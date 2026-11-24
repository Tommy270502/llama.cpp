#pragma once

// Mat-vec for the codebook types (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S).
// The grid of the type goes to SLM once per work-group, the work-groups stay resident and walk the rows, and a
// sub-group takes several rows per step so that each activation load serves all of them. A lane decodes one
// 32-weight sub-block to eight ints; codebook_apply does the dot products for all types.

#include "common.hpp"
#include "vecdotq.hpp"

// one 32-weight sub-block: d0 scales v[0..3], d1 scales v[4..7]
struct codebook_weights {
    int   v[8];
    float d0, d1;
    float m;     // IQ1_S: coefficient of the q8_1 sum
    float g[4];  // IQ1_M: coefficient of the q8_1 sum of each group of 8
};

struct codebook_activations {
    int   u[8];
    float d8, s8;
};

static __dpct_inline__ uint16_t cb_u16(const uint8_t * p) {
    return *reinterpret_cast<const uint16_t *>(p);
}

// n words from a 2-byte aligned p: n + 1 aligned dwords around them in one load, shifted into place. The window
// starts at most 4 bytes before p (inside the previous block) and ends at most 2 bytes after the last byte.
template <int n> struct cb_words {
    uint32_t w[n];
    explicit cb_words(const uint8_t * p) {
        const uintptr_t  a    = reinterpret_cast<uintptr_t>(p);
        const uint32_t * base = reinterpret_cast<const uint32_t *>((a - 2) & ~(uintptr_t) 3);
        uint32_t         d[n + 1];
#pragma unroll
        for (int k = 0; k <= n; ++k) {
            d[k] = base[k];
        }
        const bool half = (a & 2) != 0;
#pragma unroll
        for (int k = 0; k < n; ++k) {
            w[k] = half ? (d[k] >> 16) | (d[k + 1] << 16) : d[k + 1];
        }
    }
};

// grid bytes with four sign bits applied: the multiply spreads bit k to bit 8k, no carries for 4 bits
static __dpct_inline__ int cb_signed(const uint32_t grid, const uint32_t bits) {
    const uint32_t m = ((bits & 0xf) * 0x00204081u) & 0x01010101u;
    return (int) ((grid ^ (m * 0xffu)) + m);
}

template <typename T> static __dpct_inline__ void cb_fill(uint32_t * table, const T * src, int n, int id, int nthreads) {
    const uint32_t * s   = reinterpret_cast<const uint32_t *>(src);
    const int        n32 = n * (int) (sizeof(T) / sizeof(uint32_t));
    for (int i = id; i < n32; i += nthreads) {
        table[i] = s[i];
    }
}

template <ggml_type type> struct codebook;

template <> struct codebook<GGML_TYPE_IQ3_XXS> {
    static constexpr int  table_size = 256 * (int) (sizeof(uint32_t) / 4);
    static constexpr bool has_m = false, has_g = false, can_reorder = true;
    static __dpct_inline__ void fill(uint32_t * t, int id, int n) { cb_fill(t, iq3xxs_grid, 256, id, n); }

    // q3: 8 grid indices, aux: 4 x 7 sign bits + 4-bit scale
    static __dpct_inline__ codebook_weights decode(const uint8_t * q3, uint32_t aux, float d, const uint32_t * t) {
        codebook_weights w;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t sb = iq_sign_bits(aux >> (7 * l));
            w.v[2 * l + 0]    = cb_signed(t[q3[2 * l + 0]], sb);
            w.v[2 * l + 1]    = cb_signed(t[q3[2 * l + 1]], sb >> 4);
        }
        w.d0 = w.d1 = d * (0.5f + (aux >> 28)) * 0.5f;
        return w;
    }

    template <bool reordered>
    static __dpct_inline__ codebook_weights load(const uint8_t * vx, int64_t ib, int64_t nblocks, int ib32,
                                                 const uint32_t * t) {
        if constexpr (reordered) {
            const uint8_t * qs = vx + ib * (3 * QK_K / 8);
            const sycl::uint2 q = *reinterpret_cast<const sycl::uint2 *>(qs + 8 * ib32);
            uint8_t q3[8];
            *reinterpret_cast<sycl::uint2 *>(q3) = q;
            const uint32_t aux = *reinterpret_cast<const uint32_t *>(qs + QK_K / 4 + 4 * ib32);
            const float    d   = *reinterpret_cast<const ggml_half *>(vx + nblocks * (3 * QK_K / 8) + ib * 2);
            return decode(q3, aux, d, t);
        } else {
            const uint8_t *   b = vx + ib * sizeof(block_iq3_xxs);
            const cb_words<2> q(b + 2 + 8 * ib32);
            const cb_words<1> aux(b + 2 + QK_K / 4 + 4 * ib32);
            return decode(reinterpret_cast<const uint8_t *>(q.w), aux.w[0], (float) *reinterpret_cast<const ggml_half *>(b), t);
        }
    }
};

template <> struct codebook<GGML_TYPE_IQ3_S> {
    static constexpr int  table_size = 512 * (int) (sizeof(uint32_t) / 4);
    static constexpr bool has_m = false, has_g = false, can_reorder = true;
    static __dpct_inline__ void fill(uint32_t * t, int id, int n) { cb_fill(t, iq3s_grid, 512, id, n); }

    static __dpct_inline__ codebook_weights decode(const uint8_t * qs, uint32_t qh, uint32_t signs, float d, uint32_t sc,
                                                   const uint32_t * t) {
        codebook_weights w;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t s = signs >> (8 * l);
            w.v[2 * l + 0]   = cb_signed(t[(qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256))], s);
            w.v[2 * l + 1]   = cb_signed(t[(qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256))], s >> 4);
        }
        w.d0 = w.d1 = d * (1 + 2 * (sc & 0xf));
        return w;
    }

    template <bool reordered>
    static __dpct_inline__ codebook_weights load(const uint8_t * vx, int64_t ib, int64_t nblocks, int ib32,
                                                 const uint32_t * t) {
        if constexpr (reordered) {
            const sycl::uint2 q  = *reinterpret_cast<const sycl::uint2 *>(vx + ib * (QK_K / 4) + 8 * ib32);
            uint8_t           qs[8];
            *reinterpret_cast<sycl::uint2 *>(qs) = q;
            const uint32_t  qh    = vx[nblocks * (QK_K / 4) + ib * (QK_K / 32) + ib32];
            const uint8_t * sg    = vx + nblocks * (QK_K / 4 + QK_K / 32) + ib * (QK_K / 8) + 4 * ib32;
            const uint8_t * meta  = vx + nblocks * (QK_K / 4 + QK_K / 32 + QK_K / 8) + ib * (2 + IQ3S_N_SCALE);
            const float     d     = *reinterpret_cast<const ggml_half *>(meta);
            const uint32_t  sc    = meta[2 + ib32 / 2] >> (4 * (ib32 % 2));
            return decode(qs, qh, *reinterpret_cast<const uint32_t *>(sg), d, sc, t);
        } else {
            const uint8_t *   b = vx + ib * sizeof(block_iq3_s);
            const cb_words<2> q(b + offsetof(block_iq3_s, qs) + 8 * ib32);
            const uint32_t qh    = b[offsetof(block_iq3_s, qh) + ib32];
            const uint32_t signs = cb_words<1>(b + offsetof(block_iq3_s, signs) + 4 * ib32).w[0];
            const uint32_t sc    = b[offsetof(block_iq3_s, scales) + ib32 / 2] >> (4 * (ib32 % 2));
            return decode(reinterpret_cast<const uint8_t *>(q.w), qh, signs, (float) *reinterpret_cast<const ggml_half *>(b),
                          sc, t);
        }
    }
};

template <> struct codebook<GGML_TYPE_IQ2_XXS> {
    static constexpr int  table_size = 256 * (int) (sizeof(uint64_t) / 4);
    static constexpr bool has_m = false, has_g = false, can_reorder = false;
    static __dpct_inline__ void fill(uint32_t * t, int id, int n) { cb_fill(t, iq2xxs_grid, 256, id, n); }

    template <bool reordered>
    static __dpct_inline__ codebook_weights load(const uint8_t * vx, int64_t ib, int64_t, int ib32, const uint32_t * t) {
        const uint8_t * b   = vx + ib * sizeof(block_iq2_xxs);
        const uint8_t * q2  = b + 2 + 8 * ib32;
        const cb_words<2> q(q2);
        const uint32_t  idx = q.w[0];
        const uint32_t  aux = q.w[1];
        const uint64_t * t64 = reinterpret_cast<const uint64_t *>(t);
        codebook_weights w;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint64_t g  = t64[((idx >> (8 * l)) & 0xff)];
            const uint32_t sb = iq_sign_bits(aux >> (7 * l));
            w.v[2 * l + 0]    = cb_signed((uint32_t) g, sb);
            w.v[2 * l + 1]    = cb_signed((uint32_t) (g >> 32), sb >> 4);
        }
        w.d0 = w.d1 = (float) *reinterpret_cast<const ggml_half *>(b) * (0.5f + (aux >> 28)) * 0.25f;
        return w;
    }
};

template <> struct codebook<GGML_TYPE_IQ2_XS> {
    static constexpr int  table_size = 512 * (int) (sizeof(uint64_t) / 4);
    static constexpr bool has_m = false, has_g = false, can_reorder = false;
    static __dpct_inline__ void fill(uint32_t * t, int id, int n) { cb_fill(t, iq2xs_grid, 512, id, n); }

    template <bool reordered>
    static __dpct_inline__ codebook_weights load(const uint8_t * vx, int64_t ib, int64_t, int ib32, const uint32_t * t) {
        const uint8_t *  b   = vx + ib * sizeof(block_iq2_xs);
        const uint8_t *  q2  = b + offsetof(block_iq2_xs, qs) + 8 * ib32;
        const uint64_t * t64 = reinterpret_cast<const uint64_t *>(t);
        const cb_words<2> qw(q2);
        codebook_weights  w;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t q  = qw.w[l / 2] >> (16 * (l % 2)) & 0xffff;
            const uint64_t g  = t64[(q & 511)];
            const uint32_t sb = iq_sign_bits(q >> 9);
            w.v[2 * l + 0]    = cb_signed((uint32_t) g, sb);
            w.v[2 * l + 1]    = cb_signed((uint32_t) (g >> 32), sb >> 4);
        }
        const float    d  = (float) *reinterpret_cast<const ggml_half *>(b) * 0.25f;
        const uint32_t sc = b[offsetof(block_iq2_xs, scales) + ib32];
        w.d0              = d * (0.5f + (sc & 0xf));
        w.d1              = d * (0.5f + (sc >> 4));
        return w;
    }
};

template <> struct codebook<GGML_TYPE_IQ2_S> {
    static constexpr int  table_size = 1024 * (int) (sizeof(uint64_t) / 4);
    static constexpr bool has_m = false, has_g = false, can_reorder = false;
    static __dpct_inline__ void fill(uint32_t * t, int id, int n) { cb_fill(t, iq2s_grid, 1024, id, n); }

    template <bool reordered>
    static __dpct_inline__ codebook_weights load(const uint8_t * vx, int64_t ib, int64_t, int ib32, const uint32_t * t) {
        const uint8_t *  b     = vx + ib * sizeof(block_iq2_s);
        const uint32_t   idx   = cb_words<1>(b + offsetof(block_iq2_s, qs) + 4 * ib32).w[0];
        const uint32_t   signs = cb_words<1>(b + offsetof(block_iq2_s, qs) + QK_K / 8 + 4 * ib32).w[0];
        const uint32_t   qh    = b[offsetof(block_iq2_s, qh) + ib32];
        const uint64_t * t64   = reinterpret_cast<const uint64_t *>(t);
        codebook_weights w;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint64_t g = t64[(((idx >> (8 * l)) & 0xff) | ((qh << (8 - 2 * l)) & 0x300))];
            const uint32_t s = signs >> (8 * l);
            w.v[2 * l + 0]   = cb_signed((uint32_t) g, s);
            w.v[2 * l + 1]   = cb_signed((uint32_t) (g >> 32), s >> 4);
        }
        const float    d  = (float) *reinterpret_cast<const ggml_half *>(b) * 0.25f;
        const uint32_t sc = b[offsetof(block_iq2_s, scales) + ib32];
        w.d0              = d * (0.5f + (sc & 0xf));
        w.d1              = d * (0.5f + (sc >> 4));
        return w;
    }
};

template <> struct codebook<GGML_TYPE_IQ1_S> {
    static constexpr int  table_size = NGRID_IQ1S * (int) (sizeof(uint32_t) / 4);
    static constexpr bool has_m = true, has_g = false, can_reorder = false;
    static __dpct_inline__ void fill(uint32_t * t, int id, int n) { cb_fill(t, iq1s_grid_gpu, NGRID_IQ1S, id, n); }

    template <bool reordered>
    static __dpct_inline__ codebook_weights load(const uint8_t * vx, int64_t ib, int64_t, int ib32, const uint32_t * t) {
        const uint8_t * b   = vx + ib * sizeof(block_iq1_s);
        const uint32_t  idx = cb_words<1>(b + offsetof(block_iq1_s, qs) + 4 * ib32).w[0];
        const uint32_t  qh  = cb_u16(b + offsetof(block_iq1_s, qh) + 2 * ib32);
        codebook_weights w;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t g = t[(((idx >> (8 * l)) & 0xff) | (((qh >> (3 * l)) & 7) << 8))];
            w.v[2 * l + 0]   = (int) (g & 0x0f0f0f0fu);
            w.v[2 * l + 1]   = (int) ((g >> 4) & 0x0f0f0f0fu);
        }
        const float d1q = (float) *reinterpret_cast<const ggml_half *>(b) * (2 * ((qh >> 12) & 7) + 1);
        w.d0 = w.d1 = d1q;
        w.m         = d1q * (qh & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA);
        return w;
    }
};

template <> struct codebook<GGML_TYPE_IQ1_M> {
    static constexpr int  table_size = NGRID_IQ1S * (int) (sizeof(uint32_t) / 4);
    static constexpr bool has_m = false, has_g = true, can_reorder = false;
    static __dpct_inline__ void fill(uint32_t * t, int id, int n) { cb_fill(t, iq1s_grid_gpu, NGRID_IQ1S, id, n); }

    template <bool reordered>
    static __dpct_inline__ codebook_weights load(const uint8_t * vx, int64_t ib, int64_t, int ib32, const uint32_t * t) {
        const uint8_t * b   = vx + ib * sizeof(block_iq1_m);
        const uint32_t  idx = *reinterpret_cast<const uint32_t *>(b + offsetof(block_iq1_m, qs) + 4 * ib32);
        const uint32_t  qh  = cb_u16(b + offsetof(block_iq1_m, qh) + 2 * ib32);
        const uint8_t * scp = b + offsetof(block_iq1_m, scales);
        const sycl::uint2 scw = *reinterpret_cast<const sycl::uint2 *>(scp);
        const uint16_t    sc[4] = { (uint16_t) scw.x(), (uint16_t) (scw.x() >> 16), (uint16_t) scw.y(),
                                    (uint16_t) (scw.y() >> 16) };
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float f = (float) scale.f16;

        codebook_weights w;
        w.d0 = f * (2 * ((sc[ib32 / 2] >> (6 * (ib32 % 2))) & 0x7) + 1);
        w.d1 = f * (2 * ((sc[ib32 / 2] >> (6 * (ib32 % 2) + 3)) & 0x7) + 1);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t h = qh >> (4 * l);
            const uint32_t g = t[(((idx >> (8 * l)) & 0xff) | ((h & 7) << 8))];
            w.v[2 * l + 0]   = (int) (g & 0x0f0f0f0fu);
            w.v[2 * l + 1]   = (int) ((g >> 4) & 0x0f0f0f0fu);
            w.g[l]           = (h & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) * (l < 2 ? w.d0 : w.d1);
        }
        return w;
    }
};

template <bool reordered>
static __dpct_inline__ codebook_activations codebook_load_activations(const char * vy, const int ncols, const int iby) {
    codebook_activations a;
    if constexpr (reordered) {
        const sycl::int4 u0 = *reinterpret_cast<const sycl::int4 *>(vy + (size_t) iby * QK8_1);
        const sycl::int4 u1 = *reinterpret_cast<const sycl::int4 *>(vy + (size_t) iby * QK8_1 + 16);
        a.u[0] = u0.x(); a.u[1] = u0.y(); a.u[2] = u0.z(); a.u[3] = u0.w();
        a.u[4] = u1.x(); a.u[5] = u1.y(); a.u[6] = u1.z(); a.u[7] = u1.w();
        const sycl::half2 ds = *reinterpret_cast<const sycl::half2 *>(vy + ncols + (size_t) iby * sizeof(sycl::half2));
        a.d8 = ds[0];
        a.s8 = ds[1];
    } else {
        const block_q8_1 * y = reinterpret_cast<const block_q8_1 *>(vy) + iby;
        const int *        q = reinterpret_cast<const int *>(y->qs);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            a.u[k] = q[k];
        }
        a.d8 = y->ds[0];
        a.s8 = y->ds[1];
    }
    return a;
}

template <typename D>
static __dpct_inline__ float codebook_apply(const codebook_weights & w, const codebook_activations & a) {
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        s0 = dpct::dp4a(w.v[k], a.u[k], s0);
        s1 = dpct::dp4a(w.v[k + 4], a.u[k + 4], s1);
    }
    float r = w.d0 * s0 + w.d1 * s1;
    if constexpr (D::has_g) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int sy = dpct::dp4a(a.u[2 * l + 1], 0x01010101, dpct::dp4a(a.u[2 * l], 0x01010101, 0));
            r += w.g[l] * sy;
        }
    }
    r *= a.d8;
    if constexpr (D::has_m) {
        r += w.m * a.s8;
    }
    return r;
}

// Dense: rows of one matrix, ncols_dst activation columns. MoE: n_slots * nrows rows, slot s uses expert ids[s],
// activation row s (stride may be 0) and dst row s, a skipped slot (id -1) writes zeros.
template <ggml_type type, int ncols_dst, int rows_per_sg, bool reordered, bool moe>
static void mul_mat_vec_codebook(const void * __restrict__ vx_base, const void * __restrict__ vy_base,
                                 float * __restrict__ dst_base, const int32_t * __restrict__ ids, const int ncols,
                                 const int nrows, const int n_slots, const size_t stride_y_bytes,
                                 const size_t stride_dst, const size_t expert_stride, uint32_t * table,
                                 const sycl::nd_item<3> & item) {
    using D = codebook<type>;
    D::fill(table, item.get_local_linear_id(), item.get_local_range().size());
    sycl::group_barrier(item.get_group());

    const auto sg       = item.get_sub_group();
    const int  lane     = sg.get_local_linear_id();
    const int  sg_index = item.get_group_linear_id() * sg.get_group_linear_range() + sg.get_group_linear_id();
    const int  sg_count = item.get_group_range().size() * sg.get_group_linear_range();

    const int     blocks_per_row = ncols / QK_K;
    const int     ib32           = lane % (QK_K / QK8_1);
    constexpr int blocks_per_step = WARP_SIZE / (QK_K / QK8_1);
    const int     total_rows      = moe ? n_slots * nrows : nrows;

    // row0 is sub-group uniform, so the collectives below stay convergent
    for (int row0 = sg_index * rows_per_sg; row0 < total_rows; row0 += sg_count * rows_per_sg) {
        const uint8_t * vx = (const uint8_t *) vx_base;
        const char *    vy = (const char *) vy_base;
        float *         dst = dst_base;
        int             r0  = row0;
        bool            skip = false;
        if constexpr (moe) {
            const int slot = row0 / nrows;
            r0             = row0 % nrows;
            const int e    = ids[slot];
            skip           = e < 0;
            vx += (size_t) (skip ? 0 : e) * expert_stride;
            vy += (size_t) slot * stride_y_bytes;
            dst = (float *) ((char *) dst_base + (size_t) slot * stride_dst);
        }

        float sum[ncols_dst][rows_per_sg] = {};
        if (!skip) {
            const int64_t nblocks = (int64_t) nrows * blocks_per_row;
            for (int ib = lane / (QK_K / QK8_1); ib < blocks_per_row; ib += blocks_per_step) {
                codebook_weights w[rows_per_sg];
#pragma unroll
                for (int r = 0; r < rows_per_sg; ++r) {
                    const int64_t ibx = (int64_t) sycl::min(r0 + r, nrows - 1) * blocks_per_row + ib;
                    w[r]              = D::template load<reordered>(vx, ibx, nblocks, ib32, table);
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const auto a = codebook_load_activations<reordered>(
                        vy + (moe ? 0 : (size_t) j * stride_y_bytes), ncols, ib * (QK_K / QK8_1) + ib32);
#pragma unroll
                    for (int r = 0; r < rows_per_sg; ++r) {
                        sum[j][r] += codebook_apply<D>(w[r], a);
                    }
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int r = 0; r < rows_per_sg; ++r) {
                const float s = sycl::reduce_over_group(sg, sum[j][r], std::plus<>());
                if (sg.leader() && r0 + r < nrows) {
                    dst[(moe ? 0 : j * stride_dst) + r0 + r] = s;
                }
            }
        }
    }
}

template <ggml_type type, int ncols_dst, bool reordered, bool moe>
static void launch_mul_mat_vec_codebook(const void * vx, const void * vy, float * dst, const int32_t * ids,
                                        const int ncols, const int nrows, const int n_slots,
                                        const size_t stride_y_bytes, const size_t stride_dst,
                                        const size_t expert_stride, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr int rows_per_sg   = ncols_dst == 1 ? 2 : 4;
    constexpr int num_subgroups = 32;
    if constexpr (moe) {
        GGML_ASSERT(nrows % rows_per_sg == 0);
    }

    int device;
    SYCL_CHECK(CHECK_TRY_ERROR(device = get_current_device_id()));
    // nsm counts an Xe core as 16 EUs, an Xe2 core has 8
    const int xe_cores   = std::max(1, ggml_sycl_info().devices[device].nsm * 2);
    const int total_rows = moe ? n_slots * nrows : nrows;
    const int n_wg       = std::max(1, std::min(2 * xe_cores, (int) ceil_div(total_rows, num_subgroups * rows_per_sg)));

    const sycl::range<3> block_dims(1, 1, num_subgroups * WARP_SIZE);
    const sycl::range<3> block_nums(1, 1, n_wg);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint32_t, 1> table(sycl::range<1>(codebook<type>::table_size), cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_codebook<type, ncols_dst, rows_per_sg, reordered, moe>(
                                 vx, vy, dst, ids, ncols, nrows, n_slots, stride_y_bytes, stride_dst, expert_stride,
                                 table.get_multi_ptr<sycl::access::decorated::no>().get(), item);
                         });
    });
}

static constexpr bool ggml_sycl_codebook_type(const ggml_type type) {
    return type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M || type == GGML_TYPE_IQ2_XXS ||
           type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S;
}

template <ggml_type type, bool reordered>
static void mul_mat_vec_codebook_switch_ncols(const void * vx, const void * vy, float * dst, const int ncols,
                                              const int nrows, const int ncols_dst, const size_t stride_y_bytes,
                                              const size_t stride_dst, dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: launch_mul_mat_vec_codebook<type, 1, reordered, false>(vx, vy, dst, nullptr, ncols, nrows, 1, stride_y_bytes, stride_dst, 0, stream); break;
        case 2: launch_mul_mat_vec_codebook<type, 2, reordered, false>(vx, vy, dst, nullptr, ncols, nrows, 1, stride_y_bytes, stride_dst, 0, stream); break;
        case 3: launch_mul_mat_vec_codebook<type, 3, reordered, false>(vx, vy, dst, nullptr, ncols, nrows, 1, stride_y_bytes, stride_dst, 0, stream); break;
        case 4: launch_mul_mat_vec_codebook<type, 4, reordered, false>(vx, vy, dst, nullptr, ncols, nrows, 1, stride_y_bytes, stride_dst, 0, stream); break;
        case 5: launch_mul_mat_vec_codebook<type, 5, reordered, false>(vx, vy, dst, nullptr, ncols, nrows, 1, stride_y_bytes, stride_dst, 0, stream); break;
        case 6: launch_mul_mat_vec_codebook<type, 6, reordered, false>(vx, vy, dst, nullptr, ncols, nrows, 1, stride_y_bytes, stride_dst, 0, stream); break;
        case 7: launch_mul_mat_vec_codebook<type, 7, reordered, false>(vx, vy, dst, nullptr, ncols, nrows, 1, stride_y_bytes, stride_dst, 0, stream); break;
        case 8: launch_mul_mat_vec_codebook<type, 8, reordered, false>(vx, vy, dst, nullptr, ncols, nrows, 1, stride_y_bytes, stride_dst, 0, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for codebook MMVQ", ncols_dst);
    }
}

// dense entry: dispatches on the type, false for a type this file does not cover
static bool ggml_sycl_mul_mat_vec_codebook(const ggml_type type, const bool reordered, const void * vx,
                                           const void * vy, float * dst, const int ncols, const int nrows,
                                           const int ncols_dst, const size_t stride_y_bytes, const size_t stride_dst,
                                           dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 8 || ncols % QK_K != 0) {
        return false;
    }
#define CB_CASE(T)                                                                                                      \
    case T:                                                                                                             \
        if (reordered) {                                                                                                \
            if constexpr (codebook<T>::can_reorder) {                                                                   \
                mul_mat_vec_codebook_switch_ncols<T, true>(vx, vy, dst, ncols, nrows, ncols_dst, stride_y_bytes,        \
                                                           stride_dst, stream);                                         \
                return true;                                                                                            \
            }                                                                                                           \
            return false;                                                                                               \
        }                                                                                                               \
        mul_mat_vec_codebook_switch_ncols<T, false>(vx, vy, dst, ncols, nrows, ncols_dst, stride_y_bytes, stride_dst,   \
                                                    stream);                                                            \
        return true;
    switch (type) {
        CB_CASE(GGML_TYPE_IQ1_S)
        CB_CASE(GGML_TYPE_IQ1_M)
        CB_CASE(GGML_TYPE_IQ2_XXS)
        CB_CASE(GGML_TYPE_IQ2_XS)
        CB_CASE(GGML_TYPE_IQ2_S)
        CB_CASE(GGML_TYPE_IQ3_XXS)
        CB_CASE(GGML_TYPE_IQ3_S)
        default:
            return false;
    }
#undef CB_CASE
}

// MoE entry (plain layout, one activation row per slot): false for a type or shape this file does not cover
static bool ggml_sycl_mul_mat_vec_codebook_id(const ggml_type type, const void * vx, const void * vy,
                                              const int32_t * ids, float * dst, const int ncols, const int nrows,
                                              const int n_slots, const size_t expert_stride, const size_t dst_row_stride,
                                              const size_t src1_row_stride, dpct::queue_ptr stream) {
    if (ncols % QK_K != 0 || nrows % 2 != 0) {
        return false;
    }
#define CB_ID_CASE(T)                                                                                                   \
    case T:                                                                                                             \
        launch_mul_mat_vec_codebook<T, 1, false, true>(vx, vy, dst, ids, ncols, nrows, n_slots, src1_row_stride,        \
                                                       dst_row_stride, expert_stride, stream);                          \
        return true;
    switch (type) {
        CB_ID_CASE(GGML_TYPE_IQ1_S)
        CB_ID_CASE(GGML_TYPE_IQ1_M)
        CB_ID_CASE(GGML_TYPE_IQ2_XXS)
        CB_ID_CASE(GGML_TYPE_IQ2_XS)
        CB_ID_CASE(GGML_TYPE_IQ2_S)
        CB_ID_CASE(GGML_TYPE_IQ3_XXS)
        CB_ID_CASE(GGML_TYPE_IQ3_S)
        default:
            return false;
    }
#undef CB_ID_CASE
}
