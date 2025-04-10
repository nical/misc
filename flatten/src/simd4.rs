

#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
    use std::arch::x86_64 as arch;
    use arch::*;

    #[allow(non_camel_case_types)]
    pub type f32x4 = __m128;

    pub type CondMask = __m128;

    pub use arch::_mm_set1_ps as splat;
    pub use arch::_mm_setr_ps as vec4;
    pub use arch::_mm_add_ps as add;
    pub use arch::_mm_sub_ps as sub;
    pub use arch::_mm_mul_ps as mul;
    pub use arch::_mm_div_ps as div;
    pub use arch::_mm_sqrt_ps as sqrt;
    pub use arch::_mm_ceil_ps as ceil;
    pub use arch::_mm_rcp_ps as recip;
    pub use arch::_mm_min_ps as min;
    pub use arch::_mm_max_ps as max;
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
    pub use arch::_mm_load_ps as aligned_load;
    pub use arch::_mm_loadu_ps as unaligned_load;

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
    pub unsafe fn unpack_u32(lanes: f32x4) -> (u32, u32, u32, u32) {
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
    pub unsafe fn select_or_zero(cond: f32x4, a: f32x4) -> f32x4 {
        and(cond, a)
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

    #[inline(always)]
    pub unsafe fn shift_up(val: f32x4) -> f32x4 {
        const MASK: i32 = shuffle_mask(3, 0, 1, 2);
        arch::_mm_shuffle_ps::<MASK>(val, val)
    }
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64 {
    use std::arch::aarch64 as arch;

    #[allow(non_camel_case_types)]
    pub type f32x4 = arch::float32x4_t;

    #[allow(non_camel_case_types)]
    pub type u32x4 = arch::uint32x4_t;

    pub type CondMask = arch::uint32x4_t;

    pub use arch::vmovq_n_f32 as splat;
    pub use arch::vmovq_n_u32 as splat_u32;

    #[inline(always)]
    pub unsafe fn vec4(a: f32, b: f32, c: f32, d: f32) -> f32x4 {
        let v = [a, b, c, d];
        arch::vld1q_f32(v.as_ptr())
    }

    #[inline(always)]
    pub unsafe fn unpack(lanes: f32x4) -> (f32, f32, f32, f32) {
        std::mem::transmute(lanes)
    }

    #[inline(always)]
    pub unsafe fn unpack_u32(lanes: u32x4) -> (u32, u32, u32, u32) {
        std::mem::transmute(lanes)
    }


    pub use arch::vaddq_f32 as add;
    pub use arch::vsubq_f32 as sub;
    pub use arch::vmulq_f32 as mul;
    pub use arch::vdivq_f32 as div;
    pub use arch::vabsq_f32 as abs;
    pub use arch::vnegq_f32 as minus;

    pub use arch::vsqrtq_f32 as sqrt;
    pub use arch::vrndq_f32 as floor;
    //pub use arch::_mm_ceil_ps as ceil;
    pub use arch::vrecpeq_f32 as recip;
    pub use arch::vmin_f32 as min;
    pub use arch::vmax_f32 as max;
    pub use arch::vceqq_f32 as eq;
    pub use arch::vcltq_f32 as lt;
    pub use arch::vcgtq_f32 as gt;
    pub use arch::vandq_u32 as and;

    // Note: arch::vceqzq_u32 as not; is similar but leaves all
    // of the bits to 0 or 1 in a lane instead of considering
    // each bit separately.
    pub use arch::vmvnq_u32 as not;
    pub use arch::vorrq_u32 as or;
    //pub use arch::_mm_cvt_ss2si as get_first_as_int;
    pub use arch::vst1q_f32 as aligned_store;
    pub use arch::vst1q_f32 as unaligned_store;
    pub use arch::vld1q_f32 as aligned_load;
    pub use arch::vld1q_f32 as unaligned_load;


    #[inline(always)]
    pub unsafe fn mul_add(m1: f32x4, m2: f32x4, add: f32x4) -> f32x4 {
        arch::vfmaq_f32(add, m1, m2)
    }

    #[inline(always)]
    pub unsafe fn mul_sub(m1: f32x4, m2: f32x4, sub: f32x4) -> f32x4 {
        minus(arch::vfmsq_f32(sub, m1, m2))
    }

    pub use arch::vbslq_f32 as select; // (if: CondMask, then: f32x4, else: f32x4) -> f32x4

    use arch::vreinterpretq_u32_f32 as reinterpret_f32_to_u32;
    use arch::vreinterpretq_f32_u32 as reinterpret_u32_to_f32;

    #[inline(always)]
    pub unsafe fn get_first(a: f32x4) -> f32 {
        arch::vgetq_lane_f32(a, 0)
    }

    // TODO: there is probably an instruction for this?
    #[inline(always)]
    pub unsafe fn neq(a: f32x4, b: f32x4) -> CondMask {
        not(eq(a, b))
    }

    #[inline(always)]
    pub unsafe fn select_or_zero(cond: CondMask, a: f32x4) -> f32x4 {
        select(cond, a, splat(0.0))
    }

    #[inline(always)]
    pub unsafe fn is_finite(a: f32x4) -> CondMask {
        lt(abs(a), splat(f32::INFINITY))
    }

    #[inline(always)]
    pub unsafe fn sign_bit(a: f32x4) -> u32x4 {
        and(reinterpret_f32_to_u32(a), splat_u32(1u32 << 31))
    }

    #[inline(always)]
    pub unsafe fn signum(a: f32x4) -> f32x4 {
        reinterpret_u32_to_f32(
            or(sign_bit(a), splat_u32(0b00111111100000000000000000000000))
        )
    }

    #[inline(always)]
    pub unsafe fn any(a: CondMask) -> bool {
        // Ideally we would want a horitzontal or instead of this
        // horizontal add operation, so that there is no bit pattern
        // that overflows to zero. However this is well behaved when
        // all non-zero lanes are u32::MAX.
        arch::vaddvq_u32(a) != 0
    }
}

use std::mem::MaybeUninit;
#[cfg(target_arch = "x86_64")]
pub use x86_64::*;

#[cfg(target_arch = "aarch64")]
pub use aarch64::*;


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

#[repr(align(16))]
pub struct Aligned<T>(pub T);

#[repr(align(16))]
pub struct AlignedBuf(pub MaybeUninit<[f32; 16]>);

impl AlignedBuf {
    #[inline(always)]
    pub fn new() -> Self {
        AlignedBuf(MaybeUninit::uninit())
    }

    #[inline(always)]
    pub unsafe fn get(&self, offset: usize) -> f32 {
        *self.0.assume_init_ref().as_ptr().add(offset)
    }

    #[inline(always)]
    pub unsafe fn ptr(&mut self, offset: usize) -> *mut f32 {
        self.0.assume_init_mut().as_mut_ptr().add(offset)
    }

}



#[test]
pub fn sanity_check() {
    unsafe {
        const TRUE: u32 = std::u32::MAX;
        const FALSE: u32 = 0;

        fn eps_eq(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> bool {
            fn e(a: f32, b: f32) -> bool { (a - b).abs() < 0.0001 }
            e(a.0, b.0) && e(a.1, b.1) && e(a.2, b.2) && e(a.3, b.3)
        }

        assert_eq!(unpack(splat(1.0)), unpack(vec4(1.0, 1.0, 1.0, 1.0)));
        assert_eq!(unpack(vec4(1.0, 2.0, 3.0, 4.0)), (1.0, 2.0, 3.0, 4.0));
        assert_eq!(get_first(vec4(1.0, 2.0, 3.0, 4.0)), 1.0);
        assert_eq!(unpack_u32(eq(vec4(1.0, 2.0, 3.0, 4.0), vec4(2.0, 2.0, 2.0, 2.0))), (FALSE, TRUE, FALSE, FALSE));
        assert_eq!(unpack_u32(neq(vec4(1.0, 2.0, 3.0, 4.0), vec4(2.0, 2.0, 2.0, 2.0))), (TRUE, FALSE, TRUE, TRUE));
        let lt_cond = lt(vec4(1.0, 2.0, 3.0, 4.0), vec4(2.0, 2.0, 2.0, 2.0));
        let gt_cond = gt(vec4(1.0, 2.0, 3.0, 4.0), vec4(2.0, 2.0, 2.0, 2.0));
        assert_eq!(unpack_u32(lt_cond), (TRUE, FALSE, FALSE, FALSE));
        assert_eq!(unpack_u32(gt_cond), (FALSE, FALSE, TRUE, TRUE));
        assert_eq!(unpack(select(lt_cond, splat(1.0), splat(2.0))), (1.0, 2.0, 2.0, 2.0));
        assert_eq!(unpack_u32(is_finite(vec4(f32::NAN, f32::INFINITY, -f32::INFINITY, 42.0))), (FALSE, FALSE, FALSE, TRUE));
        assert_eq!(unpack(signum(vec4(0.0, -0.0, 1.0, -2.0))), (1.0, -1.0, 1.0, -1.0));
        assert!(any(eq(vec4(1.0, 1.0, 1.0, 1.0), splat(1.0))));
        assert!(any(eq(vec4(1.0, 0.0, 0.0, 0.0), splat(1.0))));
        assert!(any(eq(vec4(0.0, 1.0, 0.0, 0.0), splat(1.0))));
        assert!(any(eq(vec4(0.0, 0.0, 1.0, 0.0), splat(1.0))));
        assert!(any(eq(vec4(0.0, 0.0, 0.0, 1.0), splat(1.0))));
        assert!(!any(eq(vec4(0.0, 0.0, 0.0, 0.0), splat(1.0))));
        assert_eq!(unpack(select_or_zero(lt_cond, vec4(1.0, 2.0, 3.0, 4.0))), unpack(vec4(1.0, 0.0, 0.0, 0.0)));
        assert_eq!(unpack(abs(vec4(1.0, -1.0, -0.0, -3.1415))), (1.0, 1.0, 0.0, 3.1415));
        assert!(eps_eq(unpack(mul_add(vec4(1.0, 2.0, 3.0, 4.0), splat(2.0), splat(3.0))), (5.0, 7.0, 9.0, 11.0)));
        assert!(eps_eq(unpack(mul_sub(vec4(1.0, 2.0, 3.0, 4.0), splat(2.0), splat(3.0))), (-1.0, 1.0, 3.0, 5.0)));
        println!("sign {:?}", sign_bit(vec4(0.0, -0.0, 1.0, -2.0)));
        //println!("shift lower {:?}", shift_lower(vec4(1.0, 2.0, 3.0, 4.0)));

        let mut buf: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
        unaligned_store(buf.as_mut_ptr(), vec4(1.0, 2.0, 3.0, 4.0));
        assert_eq!(buf, [1.0, 2.0, 3.0, 4.0]);
    }
}
