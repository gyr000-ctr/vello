// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Color matrix filter implementation.

use super::FilterEffect;
use crate::filter::context::ScratchBuffer;
use vello_common::filter::color_matrix::ColorMatrix;
use vello_common::filter_effects::matrices;
use vello_common::peniko::color::PremulRgba8;
#[cfg(not(feature = "std"))]
use vello_common::peniko::kurbo::common::FloatFuncs as _;
use vello_common::pixmap::Pixmap;

impl FilterEffect for ColorMatrix {
    fn execute_lowp(&self, pixmap: &mut Pixmap, _filter_scratch: &mut ScratchBuffer) {
        apply_color_matrix(pixmap, &self.matrix);
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, _filter_scratch: &mut ScratchBuffer) {
        // The pixmap stores 8-bit premultiplied channels either way, so both
        // precision paths share the same computation.
        apply_color_matrix(pixmap, &self.matrix);
    }
}

/// Apply a 4×5 color matrix to every pixel of the pixmap.
///
/// Following the SVG `feColorMatrix` semantics, the matrix operates on
/// non-premultiplied color values: each pixel is unpremultiplied, transformed,
/// clamped to \[0, 1\] and premultiplied again.
pub(crate) fn apply_color_matrix(pixmap: &mut Pixmap, matrix: &[f32; 20]) {
    // Skip the (lossy) unpremultiply/repremultiply roundtrip for no-op matrices.
    if matrix == &matrices::IDENTITY {
        return;
    }

    for pixel in pixmap.data_mut() {
        *pixel = apply_to_pixel(pixel, matrix);
    }
}

/// Apply the color matrix to a single premultiplied pixel.
fn apply_to_pixel(pixel: &PremulRgba8, m: &[f32; 20]) -> PremulRgba8 {
    let a = f32::from(pixel.a) / 255.0;

    // Unpremultiply; a fully transparent pixel carries no color information.
    let (r, g, b) = if pixel.a == 0 {
        (0.0, 0.0, 0.0)
    } else {
        let inv_alpha = 1.0 / f32::from(pixel.a);
        // Clamp to 1.0 in case the premultiplication invariant (channel ≤ alpha)
        // is violated.
        (
            (f32::from(pixel.r) * inv_alpha).min(1.0),
            (f32::from(pixel.g) * inv_alpha).min(1.0),
            (f32::from(pixel.b) * inv_alpha).min(1.0),
        )
    };

    let out_r = (m[0] * r + m[1] * g + m[2] * b + m[3] * a + m[4]).clamp(0.0, 1.0);
    let out_g = (m[5] * r + m[6] * g + m[7] * b + m[8] * a + m[9]).clamp(0.0, 1.0);
    let out_b = (m[10] * r + m[11] * g + m[12] * b + m[13] * a + m[14]).clamp(0.0, 1.0);
    let out_a = (m[15] * r + m[16] * g + m[17] * b + m[18] * a + m[19]).clamp(0.0, 1.0);

    PremulRgba8 {
        r: (out_r * out_a * 255.0).round() as u8,
        g: (out_g * out_a * 255.0).round() as u8,
        b: (out_b * out_a * 255.0).round() as u8,
        a: (out_a * 255.0).round() as u8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::context::ScratchBuffer;
    use crate::filter::{filter_highp, filter_lowp};
    use vello_common::filter_effects::{Filter, FilterFunction, FilterGraph, FilterPrimitive};
    use vello_common::kurbo::Affine;

    const RED: PremulRgba8 = PremulRgba8 {
        r: 255,
        g: 0,
        b: 0,
        a: 255,
    };

    fn rgba(r: u8, g: u8, b: u8, a: u8) -> PremulRgba8 {
        PremulRgba8 { r, g, b, a }
    }

    /// Apply a filter to a single-pixel pixmap and return the result.
    fn apply_filter(filter: &Filter, pixel: PremulRgba8) -> PremulRgba8 {
        let mut pixmap = Pixmap::new(1, 1);
        pixmap.set_pixel(0, 0, pixel);
        let mut scratch = ScratchBuffer::new();
        filter_lowp(filter, &mut pixmap, &mut scratch, Affine::IDENTITY);
        pixmap.sample(0, 0)
    }

    fn apply_function(function: FilterFunction, pixel: PremulRgba8) -> PremulRgba8 {
        apply_filter(&Filter::from_function(function), pixel)
    }

    #[test]
    fn color_filter_grayscale() {
        // Full grayscale: luminance of red is 0.2126 * 255 ≈ 54.
        assert_eq!(
            apply_function(FilterFunction::Grayscale { amount: 1.0 }, RED),
            rgba(54, 54, 54, 255)
        );
        // Amount 0 is the identity.
        let pixel = rgba(10, 200, 130, 255);
        assert_eq!(
            apply_function(FilterFunction::Grayscale { amount: 0.0 }, pixel),
            pixel
        );
        // Partial grayscale interpolates between identity and full grayscale.
        assert_eq!(
            apply_function(FilterFunction::Grayscale { amount: 0.5 }, RED),
            rgba(155, 27, 27, 255)
        );
    }

    #[test]
    fn color_filter_sepia() {
        assert_eq!(
            apply_function(
                FilterFunction::Sepia { amount: 1.0 },
                rgba(50, 100, 150, 255)
            ),
            rgba(125, 111, 87, 255)
        );
        // Amount 0 is the identity.
        let pixel = rgba(10, 200, 130, 255);
        assert_eq!(
            apply_function(FilterFunction::Sepia { amount: 0.0 }, pixel),
            pixel
        );
    }

    #[test]
    fn color_filter_brightness() {
        let pixel = rgba(100, 200, 50, 255);
        // Zero brightness gives black, but keeps alpha.
        assert_eq!(
            apply_function(FilterFunction::Brightness { amount: 0.0 }, pixel),
            rgba(0, 0, 0, 255)
        );
        // Values above 1.0 are allowed and clamped per channel.
        assert_eq!(
            apply_function(FilterFunction::Brightness { amount: 1.5 }, pixel),
            rgba(150, 255, 75, 255)
        );
    }

    #[test]
    fn color_filter_contrast() {
        // Zero contrast gives uniform 50% gray, but keeps alpha.
        assert_eq!(
            apply_function(
                FilterFunction::Contrast { amount: 0.0 },
                rgba(100, 200, 50, 255)
            ),
            rgba(128, 128, 128, 255)
        );
        assert_eq!(
            apply_function(
                FilterFunction::Contrast { amount: 1.5 },
                rgba(100, 200, 50, 255)
            ),
            rgba(86, 236, 11, 255)
        );
    }

    #[test]
    fn color_filter_saturate() {
        // Zero saturation gives the (0.213/0.715/0.072) luminance: 0.213 * 255 ≈ 54.
        assert_eq!(
            apply_function(FilterFunction::Saturate { amount: 0.0 }, RED),
            rgba(54, 54, 54, 255)
        );
        // Oversaturation is allowed; channels are clamped.
        assert_eq!(
            apply_function(
                FilterFunction::Saturate { amount: 2.0 },
                rgba(100, 150, 200, 255)
            ),
            rgba(57, 157, 255, 255)
        );
    }

    #[test]
    fn color_filter_hue_rotate() {
        let pixel = rgba(12, 34, 220, 255);
        // Full turns (in either direction) are no-ops.
        for angle in [0.0, 360.0, -360.0, 720.0] {
            assert_eq!(
                apply_function(FilterFunction::HueRotate { angle }, pixel),
                pixel,
                "angle={angle}"
            );
        }
        // A 90° rotation of red gives (0, 0.356, 0) per the reference matrix.
        assert_eq!(
            apply_function(FilterFunction::HueRotate { angle: 90.0 }, RED),
            rgba(0, 91, 0, 255)
        );
    }

    #[test]
    fn color_filter_invert() {
        assert_eq!(
            apply_function(
                FilterFunction::Invert { amount: 1.0 },
                rgba(10, 200, 255, 255)
            ),
            rgba(245, 55, 0, 255)
        );
        // Amount 0.5 maps every channel to 0.5, keeping alpha.
        assert_eq!(
            apply_function(
                FilterFunction::Invert { amount: 0.5 },
                rgba(10, 200, 255, 255)
            ),
            rgba(128, 128, 128, 255)
        );
    }

    #[test]
    fn color_filter_opacity() {
        // Opacity scales alpha; the premultiplied color channels scale along.
        assert_eq!(
            apply_function(
                FilterFunction::Opacity { amount: 0.5 },
                rgba(200, 100, 50, 255)
            ),
            rgba(100, 50, 25, 128)
        );
        assert_eq!(
            apply_function(FilterFunction::Opacity { amount: 0.0 }, RED),
            rgba(0, 0, 0, 0)
        );
    }

    /// Out-of-range amounts are clamped like in CSS; NaN makes the filter a no-op.
    #[test]
    fn color_filter_out_of_range_amounts() {
        let pixel = rgba(10, 200, 130, 255);
        let identity = pixel;

        // Amounts above 1.0 clamp to 1.0.
        assert_eq!(
            apply_function(FilterFunction::Grayscale { amount: 5.0 }, pixel),
            apply_function(FilterFunction::Grayscale { amount: 1.0 }, pixel)
        );
        assert_eq!(
            apply_function(FilterFunction::Sepia { amount: 5.0 }, pixel),
            apply_function(FilterFunction::Sepia { amount: 1.0 }, pixel)
        );
        assert_eq!(
            apply_function(FilterFunction::Invert { amount: 5.0 }, pixel),
            apply_function(FilterFunction::Invert { amount: 1.0 }, pixel)
        );
        assert_eq!(
            apply_function(FilterFunction::Opacity { amount: 5.0 }, pixel),
            identity
        );

        // Negative amounts clamp to 0.0.
        assert_eq!(
            apply_function(FilterFunction::Grayscale { amount: -1.0 }, pixel),
            identity
        );
        assert_eq!(
            apply_function(FilterFunction::Invert { amount: -1.0 }, pixel),
            identity
        );
        assert_eq!(
            apply_function(FilterFunction::Opacity { amount: -1.0 }, pixel),
            rgba(0, 0, 0, 0)
        );
        assert_eq!(
            apply_function(FilterFunction::Brightness { amount: -1.0 }, pixel),
            apply_function(FilterFunction::Brightness { amount: 0.0 }, pixel)
        );
        assert_eq!(
            apply_function(FilterFunction::Contrast { amount: -1.0 }, pixel),
            apply_function(FilterFunction::Contrast { amount: 0.0 }, pixel)
        );
        assert_eq!(
            apply_function(FilterFunction::Saturate { amount: -1.0 }, pixel),
            apply_function(FilterFunction::Saturate { amount: 0.0 }, pixel)
        );

        // NaN makes the filter a no-op instead of erasing the layer.
        assert_eq!(
            apply_function(
                FilterFunction::Grayscale { amount: f32::NAN },
                pixel
            ),
            identity
        );
        assert_eq!(
            apply_function(FilterFunction::Opacity { amount: f32::NAN }, pixel),
            identity
        );
        assert_eq!(
            apply_function(
                FilterFunction::HueRotate { angle: f32::NAN },
                pixel
            ),
            identity
        );
    }

    /// The matrix operates on non-premultiplied values: a semi-transparent
    /// pixel must be unpremultiplied first and repremultiplied afterwards.
    #[test]
    fn color_filter_respects_premultiplied_alpha() {
        // Non-premultiplied (1.0, 0.0, 0.0) at 50% alpha, stored premultiplied.
        let pixel = rgba(128, 0, 0, 128);

        // Grayscale keeps alpha and gives 0.2126 * 128 ≈ 27 per channel.
        assert_eq!(
            apply_function(FilterFunction::Grayscale { amount: 1.0 }, pixel),
            rgba(27, 27, 27, 128)
        );

        // Opacity 0.5 halves alpha again; the color channels follow.
        assert_eq!(
            apply_function(FilterFunction::Opacity { amount: 0.5 }, pixel),
            rgba(64, 0, 0, 64)
        );

        // A fully transparent pixel has no color to transform, even for
        // matrices with a constant offset like contrast(0).
        assert_eq!(
            apply_function(
                FilterFunction::Contrast { amount: 0.0 },
                rgba(0, 0, 0, 0)
            ),
            rgba(0, 0, 0, 0)
        );
    }

    /// Primitives in a graph are applied in insertion order, each on the
    /// output of the previous one.
    #[test]
    fn color_filter_chained_primitives() {
        fn chained(primitives: &[FilterPrimitive], pixel: PremulRgba8) -> PremulRgba8 {
            let mut graph = FilterGraph::new();
            let mut last = None;
            for primitive in primitives {
                last = Some(graph.add(primitive.clone(), None));
            }
            graph.set_output(last.unwrap());
            apply_filter(
                &Filter {
                    graph: std::sync::Arc::new(graph),
                },
                pixel,
            )
        }

        let invert = FilterPrimitive::ColorMatrix {
            matrix: [
                -1.0, 0.0, 0.0, 0.0, 1.0, //
                0.0, -1.0, 0.0, 0.0, 1.0, //
                0.0, 0.0, -1.0, 0.0, 1.0, //
                0.0, 0.0, 0.0, 1.0, 0.0,
            ],
        };

        // Inverting twice restores the original.
        let pixel = rgba(10, 200, 130, 255);
        assert_eq!(chained(&[invert.clone(), invert], pixel), pixel);

        // Grayscale followed by sepia: the intermediate result is quantized to
        // 8 bits (luminance 92.98 -> 93), then sepia maps 93 to
        // (1.351, 1.203, 0.937) * 93 ≈ (126, 112, 87).
        let grayscale = FilterPrimitive::ColorMatrix {
            matrix: matrices::GRAYSCALE,
        };
        let sepia = FilterPrimitive::ColorMatrix {
            matrix: matrices::SEPIA,
        };
        assert_eq!(
            chained(&[grayscale, sepia], rgba(50, 100, 150, 255)),
            rgba(126, 112, 87, 255)
        );
    }

    /// The high-precision path shares the same 8-bit pixmap storage and must
    /// produce the same output as the low-precision path.
    #[test]
    fn color_filter_highp_matches_lowp() {
        let filter = Filter::from_function(FilterFunction::Sepia { amount: 0.7 });
        let pixel = rgba(50, 100, 150, 200);

        let mut lowp = Pixmap::new(1, 1);
        lowp.set_pixel(0, 0, pixel);
        let mut highp = Pixmap::new(1, 1);
        highp.set_pixel(0, 0, pixel);

        let mut scratch = ScratchBuffer::new();
        filter_lowp(&filter, &mut lowp, &mut scratch, Affine::IDENTITY);
        filter_highp(&filter, &mut highp, &mut scratch, Affine::IDENTITY);

        assert_eq!(lowp.sample(0, 0), highp.sample(0, 0));
    }

    /// All filter functions are supported and no longer hit `unimplemented!`.
    #[test]
    fn color_filter_all_functions_supported() {
        let functions = [
            FilterFunction::Blur { radius: 0.0 },
            FilterFunction::Brightness { amount: 1.2 },
            FilterFunction::Contrast { amount: 0.8 },
            FilterFunction::Grayscale { amount: 0.5 },
            FilterFunction::HueRotate { angle: 45.0 },
            FilterFunction::Invert { amount: 0.5 },
            FilterFunction::Opacity { amount: 0.5 },
            FilterFunction::Saturate { amount: 1.5 },
            FilterFunction::Sepia { amount: 0.5 },
        ];

        for function in functions {
            let _ = apply_function(function, rgba(50, 100, 150, 255));
        }
    }
}
