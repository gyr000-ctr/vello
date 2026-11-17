// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The color matrix filter.

/// A color matrix filter.
///
/// Applies a 4×5 matrix to the non-premultiplied RGBA values of each pixel,
/// see [`FilterPrimitive::ColorMatrix`](crate::filter_effects::FilterPrimitive::ColorMatrix).
#[derive(Debug)]
pub struct ColorMatrix {
    /// The 4×5 color transformation matrix in row-major order: 4 rows (R,G,B,A)
    /// × 5 columns (R,G,B,A,offset).
    pub matrix: [f32; 20],
}

impl ColorMatrix {
    /// Create a new color matrix filter with the given 4×5 matrix.
    pub fn new(matrix: [f32; 20]) -> Self {
        Self { matrix }
    }
}
