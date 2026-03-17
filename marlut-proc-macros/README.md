# Proc macros
This crate contains 
* Macros to automatically generate optimized versions of Lagrange extrapolations
(and interpolations). To do so we include stripped down versions of those fields for which we
plan to generate the implementation. (This is so we do not link against the main crate and create
a circular dependency.)
* The macro `lut_table_impl!`, which generates the actual implementation for each table
using the functionalities in `our_online.rs`.
