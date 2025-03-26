

pub mod x86_64 {
    use std::arch::x86_64 as arch;
    use arch::*;

    #[allow(non_camel_case_types)]
    pub type f32x4 = __m128;

    pub use arch::_mm_set1_ps as splat;
    pub use arch::_mm_setr_ps as vec4;
    pub use arch::_mm_add_ps as add;
    pub use arch::_mm_sub_ps as sub;
    pub use arch::_mm_mul_ps as mul;
    pub use arch::_mm_div_ps as div;
    pub use arch::_mm_sqrt_ps as sqrt;
    pub use arch::_mm_ceil_ps as ceil;
    pub use arch::_mm_rcp_ps as recip;
    pub use arch::_mm_cmpeq_ps as eq;
    pub use arch::_mm_cmpneq_ps as neq;
    pub use arch::_mm_cmplt_ps as lt;
    pub use arch::_mm_cmpgt_ps as gt;
    pub use arch::_mm_and_ps as and;
    pub use arch::_mm_andnot_ps as and_not;
    pub use arch::_mm_or_ps as or;
    pub use arch::_mm_cvt_ss2si as get_first_as_int;
    pub use arch::_mm_store_ps as aligned_store;
    pub use arch::_mm_storeu_ps as unaligned_store;

    //#[cfg(target_feature="fma")]
    pub use arch::_mm_fmadd_ps as mul_add;

    //#[cfg(target_feature="fma")]
    pub use arch::_mm_fmsub_ps as mul_sub;

    //#[cfg(not(target_feature="fma"))]
    //#[inline(always)]
    //pub unsafe fn mul_add(a: f32x4, b: f32x4, c: f32x4) -> f32x4 {
    //    add(mul(a, b), c)
    //}
    //#[cfg(not(target_feature="fma"))]
    //#[inline(always)]
    //pub unsafe fn mul_sub(a: f32x4, b: f32x4, c: f32x4) -> f32x4 {
    //    sub(mul(a, b), c)
    //}

    #[inline(always)]
    pub unsafe fn abs(val: f32x4) -> f32x4 {
        let minus_1 = arch::_mm_set1_epi32(-1);
        let mask = arch::_mm_castsi128_ps(arch::_mm_srli_epi32(minus_1, 1));
        arch::_mm_and_ps(mask, val)
    }

    #[inline(always)]
    pub unsafe fn unpack(lanes: f32x4) -> (f32, f32, f32, f32) {
        std::mem::transmute(lanes)
    }

    #[inline(always)]
    pub unsafe fn select(cond: f32x4, a: f32x4, b: f32x4) -> f32x4 {
        or(
            arch::_mm_andnot_ps(cond, b),
            and(cond, a),
        )
    }

    #[inline(always)]
    pub unsafe fn is_finite(a: f32x4) -> f32x4 {
        lt(abs(a), splat(f32::INFINITY))
    }

    #[inline(always)]
    pub unsafe fn sign_bit(a: f32x4) -> f32x4 {
        and(a, splat(std::mem::transmute(1u32 << 31)))
    }

    #[inline(always)]
    pub unsafe fn signum(a: f32x4) -> f32x4 {
        or(and(a, splat(std::mem::transmute(1u32 << 31))), splat(1.0))
    }

    #[inline(always)]
    pub unsafe fn not(a: f32x4) -> f32x4 {
        and_not(a, a)
    }

    pub const fn shuffle_mask(i0: i32, i1: i32, i2: i32, i3: i32) -> i32 {
        i0 | (i1 << 2) | (i2 << 4) | (i3 << 6)
    }

    #[inline(always)]
    pub unsafe fn any(a: f32x4) -> bool {
        const MASK_13: i32 = shuffle_mask(1, 0, 3, 2);
        let a13 = arch::_mm_shuffle_ps::<MASK_13>(a, a); // a1, a0, a3, a2
        let or13 =  or(a13, a); // a0|a1, a0|a1, a2|a3, a2|a3

        const MASK_02: i32 = shuffle_mask(2, 0, 0, 0);
        let a02 = arch::_mm_shuffle_ps::<MASK_02>(or13, or13); // a2|a3, a0|a1, a0|a1, a0|a1
        let or02 =  or(or13, a02);

        let val = get_first_as_int(or02);
        val != 0
    }

    #[inline(always)]
    pub unsafe fn get_first(a: f32x4) -> f32 {
        arch::_mm_cvtss_f32(a)
    }



    #[inline(always)]
    pub unsafe fn shift_lower(val: f32x4) -> f32x4 {
        const MASK: i32 = shuffle_mask(1, 2, 3, 0);
        arch::_mm_shuffle_ps::<MASK>(val, val)
    }


    #[test]
    pub fn sse_sanity_check() {
        unsafe {
            println!("eq {:?}", unpack(eq(vec4(1.0, 2.0, 3.0, 4.0), vec4(2.0, 2.0, 2.0, 2.0))));
            let lt_cond = lt(vec4(1.0, 2.0, 3.0, 4.0), vec4(2.0, 2.0, 2.0, 2.0));
            println!("lt {:?}", unpack(lt_cond));
            println!("select {:?}", unpack(select(lt_cond, splat(1.0), splat(2.0))));
            println!("is_finite {:?}", unpack(is_finite(vec4(f32::NAN, f32::INFINITY, -f32::INFINITY, 42.0))));
            println!("sign {:?}", unpack(sign_bit(vec4(0.0, -0.0, 1.0, -2.0))));
            println!("signum {:?}", unpack(signum(vec4(0.0, -0.0, 1.0, -2.0))));
            println!("any 1000 {:?}", any(vec4(1.0, 0.0, 0.0, 0.0)));
            println!("any 0100 {:?}", any(vec4(0.0, 1.0, 0.0, 0.0)));
            println!("any 0010 {:?}", any(vec4(0.0, 0.0, 1.0, 0.0)));
            println!("any 0001 {:?}", any(vec4(0.0, 0.0, 0.0, 1.0)));
            println!("any 0000 {:?}", any(vec4(0.0, 0.0, 0.0, 0.0)));
            println!("lower {:?}", get_first(vec4(1.0, 2.0, 3.0, 4.0)));
            println!("shift lower {:?}", shift_lower(vec4(1.0, 2.0, 3.0, 4.0)));
        }
    }
}

pub use x86_64::*;

#[inline(always)]
pub unsafe fn sample_cubic_horner_simd4(
    a0_x: f32x4,
    a0_y: f32x4,
    a1_x: f32x4,
    a1_y: f32x4,
    a2_x: f32x4,
    a2_y: f32x4,
    a3_x: f32x4,
    a3_y: f32x4,
    t: f32x4,
) -> (f32x4, f32x4) {
    let mut vx = a0_x;
    let mut vy = a0_y;

    vx = mul_add(a1_x, t, vx);
    vy = mul_add(a1_y, t, vy);

    let mut t2 = mul(t, t);

    vx = mul_add(a2_x, t2, vx);
    vy = mul_add(a2_y, t2, vy);

    t2 = mul(t2, t);

    vx = mul_add(a3_x, t2, vx);
    vy = mul_add(a3_y, t2, vy);

    (vx, vy)
}

#[inline(always)]
pub unsafe fn sample_cubic_horner_interlaved_simd4(
    a0: f32x4,
    a1: f32x4,
    a2: f32x4,
    a3: f32x4,
    t: f32x4,
) -> f32x4 {
    let mut v = a0;

    v = mul_add(a1, t, v);
    let mut t2 = mul(t, t);
    v = mul_add(a2, t2, v);
    t2 = mul(t2, t);
    v = mul_add(a3, t2, v);

    v
}

#[inline(always)]
pub unsafe fn sample_quadratic_horner_simd4(
    a0: f32x4,
    a1: f32x4,
    a2: f32x4,
    t: f32x4,
) -> f32x4 {
    let mut p = a0;
    p = mul_add(a1, t, p);
    p = mul_add(a2, mul(t, t), p);

    p
}

#[inline(always)]
pub unsafe fn interleave_splat(a: f32, b: f32) -> f32x4 {
    vec4(a, b, a, b)
}
