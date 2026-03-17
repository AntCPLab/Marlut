#![feature(bigint_helper_methods)]

mod mersenne61;

extern crate proc_macro;

use mersenne61::Mersenne61;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse::Parse;
use syn::{parse_macro_input, punctuated::Punctuated, LitInt, Token};
use syn::{Expr, Ident};

use derive_syn_parse::Parse;
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

trait Field:
    Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + Sum<Self>
    + Product<Self>
    + From<u64>
    + Clone
    + Copy
    + PartialEq
    + AddAssign
    + SubAssign
    + Send
    + Sync
{
    /// One the neutral element of multiplication
    const ONE: Self;

    fn inverse(self) -> Self;
}

fn compute_lagrange_denominator<Fp: Field>(num_points: usize, idx: usize) -> Fp {
    let denominator = (0..num_points)
        .map(|i| {
            if i == idx {
                Fp::ONE
            } else {
                Fp::from(idx as u64) - Fp::from(i as u64)
            }
        })
        .product::<Fp>();
    denominator.inverse()
}

fn compute_fixed_lagrange_coefficient<Fp: Field>(
    num_points: usize,
    eval_point: Fp,
    idx: usize,
) -> Fp {
    let numerator = (0..num_points)
        .map(|i| {
            if i == idx {
                Fp::ONE
            } else {
                eval_point - Fp::from(i as u64)
            }
        })
        .product::<Fp>();
    numerator * compute_lagrange_denominator(num_points, idx)
}

#[proc_macro]
pub fn mersenne61_fixed_lagrange_interpolation(
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input with Punctuated::<LitInt, Token![,]>::parse_terminated)
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(input.len(), 2);

    let num_points: usize = input[0].base10_parse().unwrap();
    let eval_point: usize = input[1].base10_parse().unwrap();
    let func_name = format_ident!(
        "mersenne61_fixed_lagrange_interpolation_{}_{}",
        num_points,
        eval_point
    );

    let eval_point = Mersenne61::from(eval_point as u64);

    let coefficients = (0..num_points)
        .map(|idx| {
            compute_fixed_lagrange_coefficient(num_points, eval_point, idx)
                .to_string()
                .parse::<TokenStream>()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let indices = (0..num_points).collect::<Vec<_>>();

    proc_macro::TokenStream::from(quote! {
        pub fn #func_name<T1: crate::share::FieldLike + std::ops::Mul<Mersenne61, Output = T1>,
            U1: crate::share::FieldLike + std::ops::Mul<Mersenne61, Output = U1>>
            (evals: &[crate::share::RssShareGeneral<T1, U1>]) -> crate::share::RssShareGeneral<T1, U1> {

            debug_assert_eq!(evals.len(), #num_points);

            #( evals[#indices] * #coefficients + )*
            crate::share::RssShareGeneral::<T1, U1>::ZERO
        }
    })
}

#[proc_macro]
pub fn mersenne61_lagrange_interpolation(
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let num_points: usize = parse_macro_input!(input as LitInt).base10_parse().unwrap();
    let func_name = format_ident!("mersenne61_lagrange_interpolation_{}", num_points);

    let denominators = (0..num_points)
        .map(|idx| {
            compute_lagrange_denominator::<Mersenne61>(num_points, idx)
                .to_string()
                .parse::<TokenStream>()
                .unwrap()
        })
        .collect::<Vec<_>>();

    let coeff_idents = (0..num_points)
        .map(|i| format_ident!("coeff_{}", i))
        .collect::<Vec<_>>();
    let create_coeff_vars = (0..num_points as u64)
        .map(|i| {
            let ident = &coeff_idents[i as usize];
            quote! {
                let #ident = eval_point - Mersenne61::from(#i);
            }
        })
        .collect::<TokenStream>();

    let prefix_idents = (0..(num_points - 1))
        .map(|i| format_ident!("prefix_{}", i))
        .collect::<Vec<_>>();
    let create_prefix_vars = (0..(num_points - 1))
        .map(|i| {
            let ident = &prefix_idents[i];
            let coeff = &coeff_idents[i];
            if i == 0 {
                quote! {
                    let #ident = #coeff;
                }
            } else {
                let prev_ident = &prefix_idents[i - 1];
                quote! {
                    let #ident = #prev_ident * #coeff;
                }
            }
        })
        .collect::<TokenStream>();

    let prepare_accumulate_suffix = quote! {
        let mut suffix = Mersenne61::ONE;
        let mut result = crate::share::RssShareGeneral::ZERO;
    };

    let accumulate_suffix = (1..num_points)
        .rev()
        .map(|i| {
            let coefficient = &denominators[i];
            let prefix_ident = &prefix_idents[i - 1];
            let coeff_ident = &coeff_idents[i];
            if i == num_points - 1 {
                quote! {
                    result += evals[#i] * #prefix_ident * #coefficient;
                    suffix *= #coeff_ident;
                }
            } else {
                quote! {
                    result += evals[#i] * suffix * #prefix_ident *  #coefficient;
                    suffix *= #coeff_ident;
                }
            }
        })
        .collect::<TokenStream>();

    let first_coefficient = &denominators[0];
    let final_suffix = quote! {
        result += evals[0] * suffix * #first_coefficient;
    };

    proc_macro::TokenStream::from(quote! {
        pub fn #func_name<T1: crate::share::FieldLike + std::ops::Mul<Mersenne61, Output = T1>,
            U1: crate::share::FieldLike + std::ops::Mul<Mersenne61, Output = U1>>
            (evals: &[crate::share::RssShareGeneral<T1, U1>], eval_point: Mersenne61) -> crate::share::RssShareGeneral<T1, U1> {

            debug_assert_eq!(evals.len(), #num_points);

            #create_coeff_vars
            #create_prefix_vars
            #prepare_accumulate_suffix
            #accumulate_suffix
            #final_suffix
            result
        }
    })
}

#[proc_macro]
pub fn mersenne61_fixed_lagrange_extrapolation(
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input with Punctuated::<LitInt, Token![,]>::parse_terminated)
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(input.len(), 2);

    let num_points: usize = input[0].base10_parse().unwrap();
    let num_extrapolated: usize = input[1].base10_parse().unwrap();
    let func_name = format_ident!(
        "mersenne61_fixed_lagrange_extrapolation_{}_{}",
        num_points,
        num_extrapolated
    );

    let indices = (num_points..num_extrapolated).collect::<Vec<_>>();
    let out_indices = (0..(num_extrapolated - num_points)).collect::<Vec<_>>();
    let interpolate_idents = (num_points..num_extrapolated).map(|i| {
        format_ident!(
            "mersenne61_fixed_lagrange_interpolation_{}_{}",
            num_points,
            i
        )
    });
    proc_macro::TokenStream::from(quote! {
        pub fn #func_name<T1: crate::share::FieldLike + std::ops::Mul<Mersenne61, Output = T1>,
            U1: crate::share::FieldLike + std::ops::Mul<Mersenne61, Output = U1>>
            (evals: &[crate::share::RssShareGeneral<T1, U1>], out: &mut [crate::share::RssShareGeneral<T1, U1>]) {

            #( marlut_proc_macros::mersenne61_fixed_lagrange_interpolation! {#num_points, #indices} )*
            #( out[#out_indices] = #interpolate_idents(evals); )*
        }
    })
}

#[proc_macro]
pub fn mersenne61_derive_lagrange_interextrapolation(
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let degree: usize = parse_macro_input!(input as LitInt).base10_parse().unwrap();
    let num_points = degree + 1;
    let target_num_points = 2 * degree + 1;

    let extrapolate_ident = format_ident!(
        "mersenne61_fixed_lagrange_extrapolation_{}_{}",
        num_points,
        target_num_points
    );
    let interpolate_ident = format_ident!("mersenne61_lagrange_interpolation_{}", num_points);
    let interpolate_target_ident =
        format_ident!("mersenne61_lagrange_interpolation_{}", target_num_points);
    proc_macro::TokenStream::from(quote! {
        impl crate::share::LagrangeInterExtrapolate<#degree> for Mersenne61 {
            fn extrapolate<T1: crate::share::FieldLike + std::ops::Mul<Self, Output = T1>,
                U1: crate::share::FieldLike + std::ops::Mul<Self, Output = U1>>
                (evals: &[crate::share::RssShareGeneral<T1, U1>], out: &mut [crate::share::RssShareGeneral<T1, U1>]) {

                marlut_proc_macros::mersenne61_fixed_lagrange_extrapolation! {#num_points, #target_num_points}
                #extrapolate_ident(evals, out)
            }

            fn interpolate<T1: crate::share::FieldLike + std::ops::Mul<Self, Output = T1>,
            U1: crate::share::FieldLike + std::ops::Mul<Self, Output = U1>>
            (evals: &[crate::share::RssShareGeneral<T1, U1>], eval_point: Self) -> crate::share::RssShareGeneral<T1, U1> {
                marlut_proc_macros::mersenne61_lagrange_interpolation! {#num_points}
                #interpolate_ident(evals, eval_point)
            }

            fn interpolate_target<T1: crate::share::FieldLike + std::ops::Mul<Self, Output = T1>,
            U1: crate::share::FieldLike + std::ops::Mul<Self, Output = U1>>
            (evals: &[crate::share::RssShareGeneral<T1, U1>], eval_point: Self) -> crate::share::RssShareGeneral<T1, U1> {
                marlut_proc_macros::mersenne61_lagrange_interpolation! {#target_num_points}
                #interpolate_target_ident(evals, eval_point)
            }
        }
    })
}

mod kw {
    syn::custom_keyword!(preshifted);
    syn::custom_keyword!(input);
    syn::custom_keyword!(output);
    syn::custom_keyword!(table);
    syn::custom_keyword!(dim_bits);
}

#[derive(Parse)]
struct PreshiftSpecifier {
    preshifted: Option<kw::preshifted>,
    #[parse_if(preshifted.is_some())]
    _comma: Option<Token![,]>,
}

#[derive(Parse)]
struct KwSpecifier<T: Parse> {
    _keyword: T,
    _sep: Token![:],
}

#[derive(Parse)]
struct LutTableImplParams {
    struct_name: Ident,

    #[prefix(Token![;])]
    preshift: PreshiftSpecifier,

    #[prefix(KwSpecifier<kw::input>)]
    input_bits: LitInt,
    _sep: Token![,],

    #[prefix(KwSpecifier<kw::output>)]
    output_bits: LitInt,
    _sep2: Token![,],

    #[prefix(KwSpecifier<kw::table>)]
    table: Expr,
    _sep3: Token![;],

    #[prefix(KwSpecifier<kw::dim_bits>)]
    #[call(Punctuated::parse_terminated)]
    dim_bits: Punctuated<LitInt, Token![,]>,
}

fn lut_table_impl_for_t_bytes(
    input: &LutTableImplParams,
    type_name: TokenStream,
    nbits: usize,
) -> TokenStream {
    let nbytes = (nbits + 7) / 8;
    let preshifted = input.preshift.preshifted.is_some();
    let input_bits: usize = input.input_bits.base10_parse().unwrap();
    let input_bytes = (input_bits + 7) / 8;
    let output_bits: usize = input.output_bits.base10_parse().unwrap();

    let table = input.table.clone();
    let dim_bits: Vec<usize> = input
        .dim_bits
        .iter()
        .map(|x| x.base10_parse().unwrap())
        .collect();
    let total_bits = dim_bits.iter().sum::<usize>()
        + ((output_bits + nbits - 1) / nbits * nbits).trailing_zeros() as usize;

    let table_ref = if preshifted {
        quote! {
            #table [*shift]
        }
    } else {
        quote! {
            #table
        }
    };

    let last_bits_per_block = 1usize << (total_bits - dim_bits.last().unwrap());
    let mut bits_per_block = vec![0; dim_bits.len()];
    *bits_per_block.last_mut().unwrap() = last_bits_per_block;
    for i in (0..dim_bits.len() - 1).rev() {
        bits_per_block[i] = bits_per_block[i + 1] / (1 << dim_bits[i]);
    }
    let elems_per_block = bits_per_block
        .iter()
        .map(|bits| *bits / nbits)
        .collect::<Vec<_>>();
    let bits_per_lookup = bits_per_block
        .iter()
        .zip(&dim_bits)
        .map(|(bytes, bits)| bytes << bits)
        .collect::<Vec<_>>();
    let elems_per_lookup = bits_per_lookup
        .iter()
        .map(|bits| *bits / nbits)
        .collect::<Vec<_>>();
    let last_bits_per_block = *bits_per_block.last().unwrap();
    let last_elems_per_block = last_bits_per_block / nbits;

    let first_dim_bits = dim_bits.last().unwrap();
    let first_dim_bits_u8 = *first_dim_bits as u8;
    let first_dim_impl = if preshifted {
        debug_assert!(last_elems_per_block >= 16);
        quote! {
            fn inner_product_first_dim(
                ohv_si: &[#type_name],
                ohv_sii: &[#type_name],
                out_si: &mut [#type_name],
                out_sii: &mut [#type_name],
                shifts: &[usize],
            ) {
                use rayon::prelude::*;
                let num_lookups = shifts.len();

                (
                    out_si.as_chunks_mut::<#last_elems_per_block>().0,
                    out_sii.as_chunks_mut::<#last_elems_per_block>().0,
                    shifts.par_iter()
                )
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, (out_si, out_sii, shift))| {
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
                        if #type_name::IS_UR && #last_elems_per_block == 16 {
                            return crate::lut_sp::our_online::inner_product_first_dim_preshifted_4::<#type_name, #output_bits, #last_elems_per_block, #first_dim_bits>(
                                ohv_si, ohv_sii, out_si, out_sii, &#table_ref, num_lookups, i
                            );
                        }
                        for j in (0.. (1 << #first_dim_bits)) {
                            let (bi, bii) = (ohv_si[j * num_lookups + i], ohv_sii[j * num_lookups + i]);
                            let offset = j * #last_elems_per_block;

                            crate::lut_sp::our_online::inner_product_first_dim_preshifted::<#type_name, #output_bits, #last_elems_per_block>(&mut *out_si, &#table_ref, bi, offset);
                            crate::lut_sp::our_online::inner_product_first_dim_preshifted::<#type_name, #output_bits, #last_elems_per_block>(&mut *out_sii, &#table_ref, bii, offset);
                        }
                    });
            }
        }
    } else {
        quote! {
            fn inner_product_first_dim(
                ohv_si: &[#type_name],
                ohv_sii: &[#type_name],
                out_si: &mut [#type_name],
                out_sii: &mut [#type_name],
                shifts: &[usize],
            ) {
                use rayon::prelude::*;
                let num_lookups = shifts.len();
                (
                    out_si.as_chunks_mut::<#last_elems_per_block>().0,
                    out_sii.as_chunks_mut::<#last_elems_per_block>().0,
                    shifts.par_iter()
                )
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(i, (out_si, out_sii, shift))| {
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
                        if #type_name::IS_UR && #input_bits == 16 && #nbits == 8 {
                            crate::lut_sp::our_online::inner_product_first_dim_ur8_65536::<#type_name, #first_dim_bits_u8>(
                                ohv_si, ohv_sii, &mut *out_si, &mut *out_sii, &#table_ref, num_lookups, i, *shift
                            );
                            return;
                        }

                        for j in (0.. (1 << #first_dim_bits)) {
                            let bi = ohv_si[j * num_lookups + i];
                            let bii = ohv_sii[j * num_lookups + i];
                            let offset = j * #last_elems_per_block;
                            #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw",))]
                            if #type_name::IS_UR && #input_bits == 16 && #nbits == 16 {
                                crate::lut_sp::our_online::inner_product_first_dim_block_opt_16_single_elem::<#type_name, #output_bits>(
                                    &mut *out_si, &mut *out_sii, &#table_ref, bi, bii, *shift, offset
                                );
                                continue;
                            }
                            crate::lut_sp::our_online::inner_product_first_dim_block::<#type_name, #output_bits>(&mut *out_si, &#table_ref, bi, *shift, offset);
                            crate::lut_sp::our_online::inner_product_first_dim_block::<#type_name, #output_bits>(&mut *out_sii, &#table_ref, bii, *shift, offset);
                        }
                    });
            }
        }
    };

    let indices = 0..dim_bits.len() - 1;

    let compute_shifts = if input_bits <= nbits {
        quote! { c.iter().map(|x| x.as_raw()).collect::<Vec<_>>()}
    } else {
        quote! {
            c.chunks_exact(#input_bytes / #nbytes)
                .map(|chunk| {
                    let mut result = 0;
                    let mut shift = 0;
                    for i in 0..chunk.len() {
                        result += chunk[i].as_raw() << shift;
                        shift += #nbits;
                    }
                    result
                })
                .collect::<Vec<_>>()
        }
    };

    let inner_product_first_dim = quote! {
        <Self as crate::lut_sp::our_online::LUT256SPTable<#type_name, Recorder, MAL>>::inner_product_first_dim(
            &ohv.ohvs.last().unwrap().e_si,
            &ohv.ohvs.last().unwrap().e_sii,
            out_ai,
            out_aii,
            &shifts,
        );
    };

    let latter_dim_ips = dim_bits[..dim_bits.len() - 1]
        .iter()
        .zip(elems_per_block.iter())
        .zip(elems_per_lookup.iter())
        .enumerate()
        .rev()
        .map(|(i, ((dim_bits, elems_per_block), elems_per_lookup))| {
            let ss_to_rss = if i == 0 {
                quote! {
                    if OUTPUT_RSS {
                        crate::lut_sp::our_online::ss_to_rss_shares(data.inner.as_party_mut(), out_ai, out_aii)?;
                        recorder.record_ip_triple_cii(&out_aii);
                    }
                }
            } else {
                quote! {
                    crate::lut_sp::our_online::ss_to_rss_shares(data.inner.as_party_mut(), out_ai, out_aii)?;
                    recorder.record_ip_triple_cii(&out_aii);
                }
            };
                quote! {
                    crate::lut_sp::our_online::inner_product_large_ss::<
                        #dim_bits,
                        #elems_per_block,
                        #elems_per_lookup,
                        #type_name,
                        _,
                    >(data.inner.as_party_mut(),
                        out_ai,
                        out_aii,
                        &ohv.ohvs[#i].e_si,
                        &ohv.ohvs[#i].e_sii,
                        recorder,
                    )?;
                    #ss_to_rss
                }
        })
        .collect::<TokenStream>();

    let dispatch_by_elems_per_lookup = |code: fn(&TokenStream, usize, usize) -> TokenStream| {
        let branches = elems_per_lookup[..elems_per_lookup.len() - 1]
            .iter()
            .zip(elems_per_block.iter())
            .map(|(elems_per_lookup, elems_per_block)| {
                let body = code(&type_name, *elems_per_lookup, *elems_per_block);
                quote! {
                    if triple.elems_per_lookup == #elems_per_lookup {
                        debug_assert_eq!(triple.elems_per_block, #elems_per_block);
                        return #body;
                    }
                }
            })
            .collect::<TokenStream>();
        quote! {
            #branches
            unreachable!("Unexpected elems per lookup");
        }
    };
    let ip_self_body = dispatch_by_elems_per_lookup(
        |type_name, elems_per_lookup, elems_per_block| {
            quote! {
                crate::lut_sp_malsec::mult_verification::process_inner_product_triple_self_impl::<#type_name, #elems_per_lookup, #elems_per_block>(triple, gammas, coeff, x1, x2)
            }
        },
    );
    let ip_next_body = dispatch_by_elems_per_lookup(
        |type_name, elems_per_lookup, elems_per_block| {
            quote! {
                crate::lut_sp_malsec::mult_verification::process_inner_product_triple_next_impl::<#type_name, #elems_per_lookup, #elems_per_block>(triple, gammas, coeff, x2)
            }
        },
    );
    let ip_prev_body = dispatch_by_elems_per_lookup(
        |type_name, elems_per_lookup, elems_per_block| {
            quote! {
                crate::lut_sp_malsec::mult_verification::process_inner_product_triple_prev_impl::<#type_name, #elems_per_lookup, #elems_per_block>(triple, gammas, coeff, x1)
            }
        },
    );

    let struct_name = input.struct_name.clone();

    quote! {
        impl crate::lut_sp::our_online::LUT256SPMalTable<#type_name> for #struct_name
        {
            fn process_inner_product_triple_self(
                triple: &crate::util::mul_triple_vec::InnerProductTriple<#type_name>,
                gammas: &[u64],
                coeff: &[u8],
                x1: &mut [crate::rep3_core::share::RssShareGeneral<crate::share::Empty, #type_name>],
                x2: &mut [crate::rep3_core::share::RssShareGeneral<#type_name, crate::share::Empty>],
            ) -> crate::rep3_core::share::RssShare<#type_name> {
                #ip_self_body
            }

            fn process_inner_product_triple_next(
                triple: &crate::util::mul_triple_vec::InnerProductTriple<#type_name>,
                gammas: &[u64],
                coeff: &[u8],
                x2: &mut [crate::rep3_core::share::RssShareGeneral<crate::share::Empty, #type_name>],
            ) -> crate::rep3_core::share::RssShareGeneral<crate::share::Empty, #type_name> {
                #ip_next_body
            }

            fn process_inner_product_triple_prev(
                triple: &crate::util::mul_triple_vec::InnerProductTriple<#type_name>,
                gammas: &[u64],
                coeff: &[u8],
                x1: &mut [crate::rep3_core::share::RssShareGeneral<#type_name, crate::share::Empty>],
            ) -> crate::rep3_core::share::RssShareGeneral<#type_name, crate::share::Empty> {
                #ip_prev_body
            }
        }

        impl<Recorder: crate::lut_sp::VerificationRecorder<#type_name>, const MAL: bool>
            crate::lut_sp::our_online::LUT256SPTable<#type_name, Recorder, MAL> for #struct_name
             {
            fn num_input_bits() -> usize {
                #input_bits
            }

            fn num_output_bits() -> usize {
                #output_bits
            }

            fn is_preshifted() -> bool {
                #preshifted
            }

            fn get_dim_bits() -> Vec<usize> {
                vec![ #(#dim_bits, )* ]
            }

            #first_dim_impl

            fn lut<const OUTPUT_RSS: bool>(
                data: &mut crate::lut_sp::LUT256SP<#type_name, Recorder, MAL>,
                v_si: &[#type_name],
                v_sii: &[#type_name],
                recorder: &mut Recorder,
            ) -> crate::rep3_core::party::error::MpcResult<(
            )> {
                use rayon::prelude::*;
                use crate::rep3_core::party::broadcast::Broadcast;
                use crate::rep3_core::share::HasZero;

                let num_lookups = v_si.len() * #nbytes / #input_bytes;

                let time = std::time::Instant::now();
                // Compute c value (shift of the table)
                let ohv = data.prep_ohv.pop().unwrap();
                let (ci, cii): (Vec<_>, Vec<_>) = (0..v_si.len())
                    .into_par_iter()
                    .map(|i| {
                        let lookup_idx = i / (#input_bytes / #nbytes);
                        let element_idx = i % (#input_bytes / #nbytes);
                        let r_idx = element_idx * num_lookups + lookup_idx;
                        (v_si[i] - ohv.prep_r_si[r_idx], v_sii[i] - ohv.prep_r_sii[r_idx])
                    })
                    .unzip();
                assert_eq!(ci.len(), v_si.len());

                let c = data.inner.as_party_mut().open_rss(&mut data.context, &ci, &cii)?;

                let shifts = {#compute_shifts};

                println!("Shift compute time {:?}", time.elapsed());

                let mut was_created = false;
                if !data.temp_vecs.is_some() {
                    data.temp_vecs = Some(crate::lut_sp::alloc_aligned_blocks(#last_bits_per_block / #nbits * num_lookups));
                    was_created = true;
                }
                let (out_ai, out_aii) = data.temp_vecs.as_mut().unwrap();
                if !was_created {
                    out_ai.fill(#type_name::ZERO);
                    out_aii.fill(#type_name::ZERO);
                    out_ai.resize(#last_bits_per_block / #nbits * num_lookups, #type_name::ZERO);
                    out_aii.resize(#last_bits_per_block / #nbits * num_lookups, #type_name::ZERO);
                }

                let time = std::time::Instant::now();
                #inner_product_first_dim

                println!("First IP time {:?}", time.elapsed());

                let time = std::time::Instant::now();
                #latter_dim_ips
                println!("IPs time {:?}", time.elapsed());

                Ok(())
            }
        }
    }
}

#[proc_macro]
pub fn lut_table_impl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as LutTableImplParams);
    let stream1 =
        lut_table_impl_for_t_bytes(&input, quote! { crate::share::unsigned_ring::UR8 }, 8);
    let stream2 =
        lut_table_impl_for_t_bytes(&input, quote! { crate::share::unsigned_ring::UR16 }, 16);
    if input.preshift.preshifted.is_some() {
        proc_macro::TokenStream::from(quote! {
            #stream1
            #stream2
        })
    } else {
        let stream3 = lut_table_impl_for_t_bytes(&input, quote! { crate::share::gf8::GF8 }, 8);
        proc_macro::TokenStream::from(quote! {
            #stream1
            #stream2
            #stream3
        })
    }
}

#[proc_macro]
pub fn lut_table_impl_boolean(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as LutTableImplParams);
    let preshifted = input.preshift.preshifted.is_some();
    let input_bits: usize = input.input_bits.base10_parse().unwrap();
    let input_bytes = (input_bits + 7) / 8;
    let output_bits: usize = input.output_bits.base10_parse().unwrap();

    let table = input.table;
    let dim_bits: Vec<usize> = input
        .dim_bits
        .iter()
        .map(|x| x.base10_parse().unwrap())
        .collect();
    let total_bits =
        dim_bits.iter().sum::<usize>() + output_bits.next_power_of_two().trailing_zeros() as usize;

    let table_ref = if preshifted {
        quote! {
            #table [*shift]
        }
    } else {
        quote! {
            #table
        }
    };

    let shift_j = if preshifted {
        quote! {}
    } else {
        quote! {
            j ^= shift;
        }
    };

    let last_bits_per_block = 1usize << (total_bits - dim_bits.last().unwrap());
    let mut bits_per_block = vec![0; dim_bits.len()];
    *bits_per_block.last_mut().unwrap() = last_bits_per_block;
    for i in (0..dim_bits.len() - 1).rev() {
        bits_per_block[i] = bits_per_block[i + 1] / (1 << dim_bits[i]);
    }
    let bytes_per_block = bits_per_block
        .iter()
        .map(|bits| (bits + 7) / 8)
        .collect::<Vec<_>>();
    let bytes_per_lookup = bytes_per_block
        .iter()
        .zip(&dim_bits)
        .map(|(bytes, bits)| bytes << bits)
        .collect::<Vec<_>>();
    let last_bytes_per_block = *bytes_per_block.last().unwrap();

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512bitalg",
        target_feature = "avx512f",
    ))]
    let has_avx512 = true;

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512bitalg",
        target_feature = "avx512f",
    )))]
    let has_avx512 = false;

    let use_first_dim_opt = has_avx512 && last_bytes_per_block == 16 && preshifted && *dim_bits.last().unwrap() == 4;

    let first_dim_impl = 
        if use_first_dim_opt {
            quote! {
                fn inner_product_first_dim(
                    ohv_si: &[crate::share::bs_bool8::BsBool8],
                    ohv_sii: &[crate::share::bs_bool8::BsBool8],
                    out_si: &mut [crate::share::bs_bool8::BsBool8],
                    out_sii: &mut [crate::share::bs_bool8::BsBool8],
                    shifts: &[usize],
                ) {
                    use std::arch::x86_64::{__m512i, _mm512_load_epi64, _mm512_mask_xor_epi64, _mm512_setzero_si512, _mm512_shuffle_i64x2, _mm512_store_epi64, _mm512_xor_epi64};
                    use std::simd::u8x64;

                    use rayon::prelude::*;

                    (
                        out_si.as_chunks_mut::<64>().0,
                        out_sii.as_chunks_mut::<64>().0,
                        ohv_si.as_chunks::<16>().0,
                        ohv_sii.as_chunks::<16>().0,
                        shifts.as_chunks::<4>().0
                    )
                        .into_par_iter()
                        .enumerate()
                        .for_each(|(i, (out_si, out_sii, ohv_si, ohv_sii, shifts))| unsafe {
                            fn merge_2(a: __m512i, b: __m512i) -> __m512i {
                                unsafe {
                                    let left = _mm512_shuffle_i64x2::<0x88>(a, b);
                                    let right = _mm512_shuffle_i64x2::<0xdd>(a, b);
                                    _mm512_xor_epi64(left, right)
                                }
                            }

                            let produce_4 = |k: usize| {
                                let mut si_0 = _mm512_setzero_si512();
                                let mut sii_0 = _mm512_setzero_si512();
    
                                let shift = &shifts[k];
                                for (j, table) in #table_ref.as_chunks::<64>().0.iter().enumerate() {
                                    let table = _mm512_load_epi64(&table[0] as *const u8 as *const i64);
                                    si_0 = _mm512_mask_xor_epi64(si_0, ohv_si[j + k * 4].0, si_0, table);
                                    sii_0 = _mm512_mask_xor_epi64(sii_0, ohv_sii[j + k * 4].0, sii_0, table);
                                }

                                let mut si_1 = _mm512_setzero_si512();
                                let mut sii_1 = _mm512_setzero_si512();

                                let shift = &shifts[k + 1];
                                for (j, table) in #table_ref.as_chunks::<64>().0.iter().enumerate() {
                                    let table = _mm512_load_epi64(&table[0] as *const u8 as *const i64);
                                    si_1 = _mm512_mask_xor_epi64(si_1, ohv_si[j + k * 4 + 4].0, si_1, table);
                                    sii_1 = _mm512_mask_xor_epi64(sii_1, ohv_sii[j + k * 4 + 4].0, sii_1, table);
                                }

                                let si_01 = merge_2(si_0, si_1);
                                let sii_01 = merge_2(sii_0, sii_1);
                                (si_01, sii_01)
                            };

                            let (si_01, sii_01) = produce_4(0);
                            let (si_23, sii_23) = produce_4(2);
                            let si = merge_2(si_01, si_23);
                            let sii = merge_2(sii_01, sii_23);

                            _mm512_store_epi64(&mut out_si[0] as *mut crate::share::bs_bool8::BsBool8 as *mut i64, si);
                            _mm512_store_epi64(&mut out_sii[0] as *mut crate::share::bs_bool8::BsBool8 as *mut i64, sii);
                        });
                }
            }
        } else {
            quote! {
                fn inner_product_first_dim(
                    ohv_si: &[crate::share::bs_bool8::BsBool8],
                    ohv_sii: &[crate::share::bs_bool8::BsBool8],
                    out_si: &mut [crate::share::bs_bool8::BsBool8],
                    out_sii: &mut [crate::share::bs_bool8::BsBool8],
                    shifts: &[usize],
                ) {
                    use rayon::prelude::*;
                    let ohv_block_size = (shifts.len() + 7) / 8;

                    (
                        out_si.as_chunks_mut::<#last_bytes_per_block>().0,
                        out_sii.as_chunks_mut::<#last_bytes_per_block>().0,
                        shifts.par_iter()
                    )
                        .into_par_iter()
                        .enumerate()
                        .for_each(|(i, (out_si, out_sii, shift))| {
                            let byte_index = i / 8;
                            let mask = 1 << ( i % 8);
                            for (mut j, table) in #table_ref.as_chunks::<#last_bytes_per_block>().0.iter().enumerate() {
                                #shift_j
                                let bi = (ohv_si[j * ohv_block_size + byte_index].0 & mask) != 0;
                                let bii = (ohv_sii[j * ohv_block_size + byte_index].0 & mask) != 0;
                                let table_bsbool = unsafe {
                                    &*(table as *const [u8] as *const [crate::share::bs_bool8::BsBool8])
                                };
                                if bi {
                                    crate::lut_sp_boolean::our_online::xor_block(&mut *out_si, table_bsbool);
                                }
                                if bii {
                                    crate::lut_sp_boolean::our_online::xor_block(&mut *out_sii, table_bsbool);
                                }
                            }
                        });
                }
            }
    };

    let indices = 0..dim_bits.len() - 1;

    let compute_shifts = if preshifted {
        if input_bits <= 8 {
            quote! { c.iter().map(|x| x.0 as usize).collect::<Vec<_>>() }
        } else {
            quote! {
                c.chunks_exact(#input_bytes)
                    .map(|chunk| {
                        let mut result = 0;
                        let mut shift = 0;
                        for i in 0..chunk.len() {
                            result += (chunk[i].0 as usize) << shift;
                            shift += 8;
                        }
                        result
                    })
                    .collect::<Vec<_>>()
            }
        }
    } else {
        let num_dims = dim_bits.len();
        let mut body = quote! {
            let mut shifts = Vec::with_capacity(#num_dims * num_lookups);
        };
        let mut num_bits = 0;
        for dim_bit in &dim_bits {
            if input_bits == 8 {
                let mask = (((1 << dim_bit) - 1) << num_bits) as u8;
                body.extend(quote! {
                    shifts.extend(c.iter().map(|c| ((c.0 & #mask) >> #num_bits) as usize));
                });
            } else {
                body.extend(quote! {
                    shifts.extend(c.chunks_exact(#input_bytes).map(|c_chunk| {
                        crate::lut_sp_boolean::our_online::compute_shift(&c_chunk, #num_bits, #dim_bit)
                    }));
                })
            }
            num_bits += dim_bit;
        }
        body.extend(quote! { shifts });
        quote! {
            {#body}
        }
    };

    let slice_last_ohv = if use_first_dim_opt {
        quote! {
            let (last_ohv_si, last_ohv_sii) = rayon::join(
                || crate::lut_sp_boolean::our_offline::slice_and_double_16(&ohv.ohvs.last().unwrap().e_si),
                || crate::lut_sp_boolean::our_offline::slice_and_double_16(&ohv.ohvs.last().unwrap().e_sii),
            );
        }
    } else {
        quote! {}
    };

    let slice_first_ohv = if output_bits % 8 != 0 || output_bits == 8 {
        if preshifted {
            quote! {
                let (sliced_ohv_si, sliced_ohv_sii) = rayon::join(
                    || crate::lut_sp_boolean::our_offline::slice_ohv(&ohv.ohvs[0].e_si, num_lookups),
                    || crate::lut_sp_boolean::our_offline::slice_ohv(&ohv.ohvs[0].e_sii, num_lookups),
                );
            }
        } else {
            quote! {
                let (sliced_ohv_si, sliced_ohv_sii) = rayon::join(
                    || crate::lut_sp_boolean::our_offline::slice_and_shift_ohv(&ohv.ohvs[0].e_si, &shifts[..num_lookups]),
                    || crate::lut_sp_boolean::our_offline::slice_and_shift_ohv(&ohv.ohvs[0].e_sii, &shifts[..num_lookups]),
                );
            }
        }
    } else {
        quote! {}
    };

    let inner_product_first_dim = if use_first_dim_opt {
        quote! {
            <Self as crate::lut_sp_boolean::our_online::LUT256SPTable<Recorder, MAL>>::inner_product_first_dim(
                &last_ohv_si,
                &last_ohv_sii,
                out_ai,
                out_aii,
                &shifts,
            );
        }
    } else if preshifted {
        quote! {
            <Self as crate::lut_sp_boolean::our_online::LUT256SPTable<Recorder, MAL>>::inner_product_first_dim(
                &ohv.ohvs.last().unwrap().e_si,
                &ohv.ohvs.last().unwrap().e_sii,
                out_ai,
                out_aii,
                &shifts,
            );
        }
    } else {
        quote! {
            <Self as crate::lut_sp_boolean::our_online::LUT256SPTable<Recorder, MAL>>::inner_product_first_dim(
                &ohv.ohvs.last().unwrap().e_si,
                &ohv.ohvs.last().unwrap().e_sii,
                out_ai,
                out_aii,
                &shifts[shifts.len() - num_lookups..],
            );
        }
    };

    let latter_dim_ips = 
        dim_bits[..dim_bits.len() - 1]
        .iter()
        .zip(bytes_per_block.iter())
        .zip(bytes_per_lookup.iter())
        .enumerate()
        .rev()
        .map(|(i, ((dim_bits, bytes_per_block), bytes_per_lookup))| {
            let ss_to_rss = if i == 0 {
                quote! {
                    if OUTPUT_RSS {
                        crate::lut_sp_boolean::our_online::ss_to_rss_shares(data.inner.as_party_mut(), out_ai, out_aii)?;
                        recorder.record_ip_triple_cii(out_aii);
                    }
                }
            } else {
                quote! {
                    crate::lut_sp_boolean::our_online::ss_to_rss_shares(data.inner.as_party_mut(), out_ai, out_aii)?;
                    recorder.record_ip_triple_cii(out_aii);
                }
            };
            if i == 0   && output_bits % 8 != 0 {
                quote! {
                    crate::lut_sp_boolean::our_online::inner_product_small_ss(
                        data.inner.as_party_mut(),
                        out_ai,
                        out_aii,
                        &sliced_ohv_si,
                        &sliced_ohv_sii,
                        #dim_bits,
                        num_lookups,
                    )?;
                    #ss_to_rss
                }
            } else if has_avx512 && i == 0 && output_bits == 8 {
                quote! {
                    crate::lut_sp_boolean::our_online::inner_product_large_ss_opt_byte::<
                        #dim_bits,
                        #bytes_per_lookup,
                        #preshifted,
                    >(
                        data.inner.as_party_mut(),
                        out_ai,
                        out_aii,
                        &sliced_ohv_si,
                        &sliced_ohv_sii,
                        &ohv.ohvs[0].e_si,
                        &ohv.ohvs[0].e_sii,
                        &shifts[..num_lookups],
                        recorder,
                    )?;
                    #ss_to_rss
                }
            } else if preshifted {
                quote! {
                    crate::lut_sp_boolean::our_online::inner_product_large_ss::<
                        #dim_bits,
                        #bytes_per_block,
                        #bytes_per_lookup,
                        #preshifted,
                    >(data.inner.as_party_mut(),
                        out_ai,
                        out_aii,
                        &ohv.ohvs[#i].e_si,
                        &ohv.ohvs[#i].e_sii,
                        &shifts,
                        recorder,
                    )?;
                    #ss_to_rss
                }
            } else {
                quote! {
                    crate::lut_sp_boolean::our_online::inner_product_large_ss::<
                        #dim_bits,
                        #bytes_per_block,
                        #bytes_per_lookup,
                        #preshifted,
                    >(data.inner.as_party_mut(),
                        out_ai,
                        out_aii,
                        &ohv.ohvs[#i].e_si,
                        &ohv.ohvs[#i].e_sii,
                        &shifts[#i * num_lookups..(#i + 1) * num_lookups],
                        recorder,
                    )?;
                    #ss_to_rss
                }
            }
        })
        .collect::<TokenStream>();

    let struct_name = input.struct_name;
    proc_macro::TokenStream::from(quote! {
        impl<Recorder: crate::lut_sp_boolean::VerificationRecorder, const MAL: bool>
        crate::lut_sp_boolean::our_online::LUT256SPTable<Recorder, MAL> for #struct_name {
            fn num_input_bits() -> usize {
                #input_bits
            }

            fn num_output_bits() -> usize {
                #output_bits
            }

            fn is_preshifted() -> bool {
                #preshifted
            }

            fn get_dim_bits() -> Vec<usize> {
                vec![ #(#dim_bits, )* ]
            }

            #first_dim_impl

            fn lut<const OUTPUT_RSS: bool>(
                data: &mut crate::lut_sp_boolean::LUT256SP<Recorder, MAL>,
                v_si: &[crate::share::bs_bool8::BsBool8],
                v_sii: &[crate::share::bs_bool8::BsBool8],
                recorder: &mut Recorder,
            ) -> crate::rep3_core::party::error::MpcResult<()> {
                use rayon::prelude::*;
                use crate::rep3_core::party::broadcast::Broadcast;
                use crate::rep3_core::share::HasZero;

                let num_lookups = v_si.len() / #input_bytes;

                let time = std::time::Instant::now();
                // Compute c value (shift of the table)
                let ohv = data.prep_ohv.pop().unwrap();
                let (ci, cii): (Vec<_>, Vec<_>) = (
                    ohv.prep_r_si.par_iter(),
                    ohv.prep_r_sii.par_iter(),
                    v_si,
                    v_sii,
                )
                    .into_par_iter()
                    .map(|(r_si, r_sii, v_si, v_sii)| (*r_si + *v_si, *r_sii + *v_sii))
                    .unzip();
                assert_eq!(ci.len(), v_si.len());
                let c = data.inner.as_party_mut().open_rss(&mut data.context, &ci, &cii)?;

                let shifts = #compute_shifts;

                println!("Shift compute time {:?}", time.elapsed());

                #slice_last_ohv
                #slice_first_ohv

                let mut was_created = false;
                if !data.temp_vecs.is_some() {
                    data.temp_vecs = Some(crate::lut_sp_boolean::alloc_aligned_blocks(#last_bytes_per_block * num_lookups));
                    was_created = true;
                }
                let (out_ai, out_aii) = data.temp_vecs.as_mut().unwrap();
                if !was_created {
                    out_ai.fill(crate::share::bs_bool8::BsBool8::ZERO);
                    out_aii.fill(crate::share::bs_bool8::BsBool8::ZERO);
                    out_ai.resize(#last_bytes_per_block * num_lookups, crate::share::bs_bool8::BsBool8::ZERO);
                    out_aii.resize(#last_bytes_per_block * num_lookups, crate::share::bs_bool8::BsBool8::ZERO);
                }

                let time = std::time::Instant::now();
                #inner_product_first_dim

                println!("First IP time {:?}", time.elapsed());

                let time = std::time::Instant::now();
                #latter_dim_ips
                println!("IPs time {:?}", time.elapsed());

                Ok(())
            }
        }
    })
}
