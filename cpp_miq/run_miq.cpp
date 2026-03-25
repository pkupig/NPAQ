/**
 * run_miq  —  Standalone Mixed-Integer Quadrangulation runner.
 *
 * Pipeline:
 *   1. Read triangle mesh.
 *   2. Read per-vertex GL cross-field  u_real.txt / u_imag.txt
 *      (u_i = exp(4iθ_i), per-vertex complex, produced by the Python GL solver).
 *   3. Compute per-face mismatches directly from u using parallel transport.
 *      This bypasses the noisy PD1/PD2 → bisector → mismatch conversion and
 *      avoids the spurious singularities that caused 200+ UV fold-overs.
 *   4. Find singularities + Poincaré–Hopf check.
 *   5. Cut mesh from singularities.
 *   6. Comb the (optional) per-face PD1/PD2 frame field guided by mismatch.
 *   7. Run igl::copyleft::comiso::miq.
 *   8. Extract quads via barycentric UV sampling (not nearest-vertex snapping).
 *
 * Usage:
 *   ./run_miq  <mesh.obj>  <u_real.txt>  <u_imag.txt>  <out_quad.obj>
 *              [gradient_size=20]  [stiffness=5]  [direct_round=1]  [iter=5]
 *              [pd1.txt]  [pd2.txt]   ← optional; used only for comb_frame_field
 *
 * Build:
 *   cd cpp_miq && mkdir -p build && cd build && cmake .. && make -j4
 */

#include <Eigen/Core>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/comb_cross_field.h>
#include <igl/comb_frame_field.h>
#include <igl/cross_field_mismatch.h>
#include <igl/find_cross_field_singularities.h>
#include <igl/cut_mesh_from_singularities.h>
#include <igl/copyleft/comiso/miq.h>
#include <igl/compute_frame_field_bisectors.h>

#ifdef HAS_LIBQEX
#include <qex.h>         // libQEx C interface (interfaces/c/qex.h)
#endif

#include <array>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct BoundaryStats {
    int boundary_edges = 0;
    int boundary_loops = 0;
    int boundary_chains = 0;
    int irregular_boundary_vertices = 0;
};

// ─── I/O helpers ──────────────────────────────────────────────────────────────

static bool read_matrix_txt(const std::string& path, Eigen::MatrixXd& M)
{
    std::ifstream in(path);
    if (!in.is_open()) { std::cerr << "[run_miq] Cannot open: " << path << "\n"; return false; }
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::vector<double> row; double v;
        while (ss >> v) row.push_back(v);
        if (!row.empty()) rows.push_back(row);
    }
    if (rows.empty()) return false;
    int nr = (int)rows.size(), nc = (int)rows[0].size();
    M.resize(nr, nc);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < std::min(nc,(int)rows[i].size()); ++j)
            M(i,j) = rows[i][j];
    return true;
}

static bool read_vector_txt(const std::string& path, Eigen::VectorXd& v)
{
    Eigen::MatrixXd M;
    if (!read_matrix_txt(path, M)) return false;
    v = M.col(0);
    return true;
}

// ─── Per-vertex normals ───────────────────────────────────────────────────────

static Eigen::MatrixXd compute_vertex_normals(
    const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    // Use UNIFORM (not area-weighted) face normal averaging to match Python's
    // compute_vertex_frames which normalises each face normal before accumulating.
    Eigen::MatrixXd VN = Eigen::MatrixXd::Zero(V.rows(), 3);
    for (int f = 0; f < F.rows(); ++f) {
        Eigen::Vector3d a = V.row(F(f,1)) - V.row(F(f,0));
        Eigen::Vector3d b = V.row(F(f,2)) - V.row(F(f,0));
        Eigen::Vector3d fn = a.cross(b);
        double nn = fn.norm();
        if (nn > 1e-10) fn /= nn;   // ← unit normal per face (uniform weight)
        VN.row(F(f,0)) += fn; VN.row(F(f,1)) += fn; VN.row(F(f,2)) += fn;
    }
    for (int i = 0; i < (int)VN.rows(); ++i) {
        double n = VN.row(i).norm();
        if (n > 1e-10) VN.row(i) /= n;
    }
    return VN;
}

// ─── Duff et al. (2017) per-vertex tangent frames ────────────────────────────
//
//  Builds a smooth, C¹-continuous orthonormal tangent frame (e1, e2) for every
//  vertex from its unit normal.  This is the same formula used by the Python GL
//  solver (crossfield.py), ensuring the transport angles match.

static void compute_duff_frames(
    const Eigen::MatrixXd& VN,
    Eigen::MatrixXd& E1,   // N×3
    Eigen::MatrixXd& E2)   // N×3
{
    int N = (int)VN.rows();
    E1.resize(N, 3); E2.resize(N, 3);
    for (int i = 0; i < N; ++i) {
        double nx = VN(i,0), ny = VN(i,1), nz = VN(i,2);
        double sign = (nz >= 0.0) ? 1.0 : -1.0;
        double a = -1.0 / (sign + nz);
        double b = nx * ny * a;
        E1.row(i) << 1.0 + sign*nx*nx*a,  sign*b,  -sign*nx;
        E2.row(i) << b,  sign + ny*ny*a,  -ny;
        E1.row(i) /= E1.row(i).norm();
        E2.row(i) /= E2.row(i).norm();
    }
}

// ─── Per-face directions from per-vertex u (complex averaging) ────────────────
//
//  Converts per-vertex GL cross-field u = exp(4iθ) to per-face principal
//  directions PD1, PD2, using complex-space averaging in each face's tangent
//  plane.  This is the correct approach for 4-RoSy fields:
//
//    For each face f with vertices i,j,k:
//      1. Compute face tangent frame (fe1, fe2, fn).
//      2. For each vertex, map its direction d = cos(θ)*e1 + sin(θ)*e2
//         to an angle alpha in the face frame: alpha = atan2(d·fe2, d·fe1).
//      3. Accumulate in exp(4i·alpha) space: w_f += exp(4i·alpha).
//         (Complex averaging respects 4-fold symmetry — no branch disambiguation.)
//      4. theta_f = arg(w_f) / 4.
//      5. PD1_f = cos(theta_f)*fe1 + sin(theta_f)*fe2.
//         PD2_f = -sin(theta_f)*fe1 + cos(theta_f)*fe2.
//
//  After this, call igl::cross_field_mismatch(V, F, PD1, PD2, false, MMatch)
//  to get the mismatch matrix using igl's own transport convention — this
//  guarantees consistency with igl::find_cross_field_singularities and miq.

static void compute_face_directions_from_u(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& u_real,
    const Eigen::VectorXd& u_imag,
    const Eigen::MatrixXd& E1,   // per-vertex Duff e1  (N×3)
    const Eigen::MatrixXd& E2,   // per-vertex Duff e2  (N×3)
    Eigen::MatrixXd& PD1,        // output (F×3)
    Eigen::MatrixXd& PD2)        // output (F×3)
{
    int nF = (int)F.rows();
    PD1.resize(nF, 3);
    PD2.resize(nF, 3);

    for (int f = 0; f < nF; ++f) {
        // ── Face tangent frame ─────────────────────────────────────────────
        Eigen::Vector3d ea = (V.row(F(f,1)) - V.row(F(f,0))).transpose();
        Eigen::Vector3d eb = (V.row(F(f,2)) - V.row(F(f,0))).transpose();
        Eigen::Vector3d fn = ea.cross(eb);
        double fnn = fn.norm();
        if (fnn < 1e-10) { fn = Eigen::Vector3d(0,0,1); } else { fn /= fnn; }

        // fe1 along first edge, projected out of fn
        Eigen::Vector3d fe1 = ea - ea.dot(fn)*fn;
        double fe1n = fe1.norm();
        if (fe1n < 1e-10) {
            // Degenerate: choose any perpendicular to fn
            fe1 = fn.cross(Eigen::Vector3d(1,0,0));
            if (fe1.norm() < 1e-6) fe1 = fn.cross(Eigen::Vector3d(0,1,0));
        }
        fe1.normalize();
        Eigen::Vector3d fe2 = fn.cross(fe1);

        // ── Complex averaging with normal-alignment weighting ─────────────
        //
        // Weight each vertex's contribution by |dot(vertex_normal, face_normal)|.
        // Vertices whose normals are nearly perpendicular to the face normal have
        // their direction projection ill-conditioned and introduce noise.
        double wx = 0.0, wy = 0.0, wtot = 0.0;
        for (int k = 0; k < 3; ++k) {
            int i = F(f, k);
            // Compute vertex normal n_i = cross(E1_i, E2_i) from Duff frames
            Eigen::Vector3d e1i = E1.row(i).transpose();
            Eigen::Vector3d e2i = E2.row(i).transpose();
            Eigen::Vector3d n_i = e1i.cross(e2i);
            double w_i = std::max(0.0, n_i.dot(fn));   // alignment weight ∈ [0,1]
            if (w_i < 1e-6) continue;                   // skip nearly-perpendicular vertices

            double theta_i = std::atan2(u_imag(i), u_real(i)) / 4.0;
            // 3D direction in vertex i's Duff frame
            Eigen::Vector3d d = std::cos(theta_i)*E1.row(i).transpose()
                              + std::sin(theta_i)*E2.row(i).transpose();
            // Angle of d in face tangent frame
            double alpha_i = std::atan2(d.dot(fe2), d.dot(fe1));
            // Accumulate in exp(4i·alpha) space, weighted by normal alignment
            wx += w_i * std::cos(4.0 * alpha_i);
            wy += w_i * std::sin(4.0 * alpha_i);
            wtot += w_i;
        }
        if (wtot < 1e-10) {
            // All vertices nearly perpendicular to face — use unweighted fallback
            for (int k = 0; k < 3; ++k) {
                int i = F(f, k);
                double theta_i = std::atan2(u_imag(i), u_real(i)) / 4.0;
                Eigen::Vector3d d = std::cos(theta_i)*E1.row(i).transpose()
                                  + std::sin(theta_i)*E2.row(i).transpose();
                double alpha_i = std::atan2(d.dot(fe2), d.dot(fe1));
                wx += std::cos(4.0 * alpha_i);
                wy += std::sin(4.0 * alpha_i);
            }
        }

        double theta_f = std::atan2(wy, wx) / 4.0;
        PD1.row(f) = (std::cos(theta_f)*fe1 + std::sin(theta_f)*fe2).transpose();
        PD2.row(f) = (-std::sin(theta_f)*fe1 + std::cos(theta_f)*fe2).transpose();
    }
}

// ─── Poincaré–Hopf verification ───────────────────────────────────────────────

static void check_poincare_hopf(
    const Eigen::MatrixXi& F,
    int n_verts,
    const Eigen::Matrix<int, Eigen::Dynamic, 1>& singularityIndex)
{
    std::set<std::pair<int,int>> edges;
    for (int f = 0; f < F.rows(); ++f)
        for (int k = 0; k < 3; ++k) {
            int a = F(f,k), b = F(f,(k+1)%3);
            if (a>b) std::swap(a,b);
            edges.insert({a,b});
        }
    int chi = n_verts - (int)edges.size() + (int)F.rows();
    int sum_idx = singularityIndex.sum();
    int expected = 4 * chi;

    std::cout << "[run_miq] Poincaré–Hopf: Σ singularity_index = " << sum_idx
              << "  expected = " << expected << "  (χ = " << chi << ")\n";
    if (std::abs(sum_idx - expected) > 0)
        std::cerr << "[run_miq] WARNING: P-H violated by "
                  << std::abs(sum_idx - expected) << " units.\n";
    else
        std::cout << "[run_miq] Poincaré–Hopf satisfied ✓\n";
}

// ─── Quad winding fix ────────────────────────────────────────────────────────

static void fix_quad_winding(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& VN,
    Eigen::MatrixXi& quadF)
{
    int flipped = 0;
    for (int i = 0; i < (int)quadF.rows(); ++i) {
        int v0=quadF(i,0), v1=quadF(i,1), v2=quadF(i,2), v3=quadF(i,3);
        Eigen::Vector3d d1 = V.row(v2) - V.row(v0);
        Eigen::Vector3d d2 = V.row(v3) - V.row(v1);
        Eigen::Vector3d qn = d1.cross(d2);
        Eigen::Vector3d avg_n = (VN.row(v0)+VN.row(v1)+VN.row(v2)+VN.row(v3)).transpose();
        if (qn.dot(avg_n) < 0.0) {
            std::swap(quadF(i,1), quadF(i,3));
            ++flipped;
        }
    }
    if (flipped > 0)
        std::cout << "[run_miq] Winding corrected: " << flipped << "/" << quadF.rows() << "\n";
}

// ─── Boundary diagnostics / slit-seam welding ────────────────────────────────

static std::vector<std::vector<int>> extract_boundary_loops_quad(const Eigen::MatrixXi& F)
{
    using Edge = std::pair<int,int>;
    std::map<Edge, int> edge_counts;
    for (int i = 0; i < (int)F.rows(); ++i) {
        for (int k = 0; k < 4; ++k) {
            int a = F(i,k), b = F(i,(k+1)%4);
            if (a > b) std::swap(a,b);
            edge_counts[{a,b}] += 1;
        }
    }

    std::vector<Edge> boundary_edges;
    for (const auto& kv : edge_counts)
        if (kv.second == 1) boundary_edges.push_back(kv.first);

    std::map<int, std::vector<int>> adj;
    for (const auto& e : boundary_edges) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
    }

    std::set<Edge> seen;
    std::vector<std::vector<int>> loops;
    for (const auto& e : boundary_edges) {
        if (seen.count(e) || seen.count({e.second, e.first})) continue;
        if ((int)adj[e.first].size() != 2 || (int)adj[e.second].size() != 2) continue;

        int start = e.first;
        int prev = e.first;
        int cur = e.second;
        std::vector<int> loop{start, cur};
        seen.insert(e);
        seen.insert({e.second, e.first});

        while (cur != start) {
            const auto& nbrs = adj[cur];
            int nxt = -1;
            for (int nb : nbrs) {
                if (nb != prev) { nxt = nb; break; }
            }
            if (nxt < 0) break;
            prev = cur;
            cur = nxt;
            if (cur != start) {
                if (std::find(loop.begin(), loop.end(), cur) != loop.end()) break;
                loop.push_back(cur);
            }
            Edge ek = (prev < cur) ? Edge(prev, cur) : Edge(cur, prev);
            if (seen.count(ek)) break;
            seen.insert(ek);
            seen.insert({ek.second, ek.first});
        }
        if (cur == start && (int)loop.size() >= 3) loops.push_back(loop);
    }
    return loops;
}

static BoundaryStats boundary_stats_quad(const Eigen::MatrixXi& F)
{
    using Edge = std::pair<int,int>;
    std::map<Edge, int> edge_counts;
    for (int i = 0; i < (int)F.rows(); ++i) {
        for (int k = 0; k < 4; ++k) {
            int a = F(i,k), b = F(i,(k+1)%4);
            if (a > b) std::swap(a,b);
            edge_counts[{a,b}] += 1;
        }
    }

    std::map<int, std::set<int>> adj;
    BoundaryStats out;
    for (const auto& kv : edge_counts) {
        if (kv.second != 1) continue;
        ++out.boundary_edges;
        adj[kv.first.first].insert(kv.first.second);
        adj[kv.first.second].insert(kv.first.first);
    }
    for (const auto& kv : adj)
        if ((int)kv.second.size() != 2) ++out.irregular_boundary_vertices;

    std::set<int> seen;
    for (const auto& kv : adj) {
        int seed = kv.first;
        if (seen.count(seed)) continue;
        std::vector<int> stack{seed};
        std::vector<int> comp;
        seen.insert(seed);
        while (!stack.empty()) {
            int cur = stack.back();
            stack.pop_back();
            comp.push_back(cur);
            for (int nb : adj[cur]) {
                if (!seen.count(nb)) {
                    seen.insert(nb);
                    stack.push_back(nb);
                }
            }
        }
        bool is_loop = true;
        for (int v : comp) if ((int)adj[v].size() != 2) { is_loop = false; break; }
        if (is_loop) ++out.boundary_loops;
        else ++out.boundary_chains;
    }
    return out;
}

static void compress_quad_mesh(
    Eigen::MatrixXd& V,
    Eigen::MatrixXd& VN,
    Eigen::MatrixXi& F)
{
    std::set<int> used;
    for (int i = 0; i < (int)F.rows(); ++i)
        for (int k = 0; k < 4; ++k)
            used.insert(F(i,k));

    std::map<int,int> remap;
    Eigen::MatrixXd V2((int)used.size(), 3);
    Eigen::MatrixXd VN2((int)used.size(), 3);
    int idx = 0;
    for (int old : used) {
        remap[old] = idx;
        V2.row(idx) = V.row(old);
        VN2.row(idx) = VN.row(old);
        ++idx;
    }
    for (int i = 0; i < (int)F.rows(); ++i)
        for (int k = 0; k < 4; ++k)
            F(i,k) = remap[F(i,k)];
    V = V2;
    VN = VN2;
}

static bool merge_vertex_pair_trial(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& VN,
    const Eigen::MatrixXi& F,
    int va,
    int vb,
    Eigen::MatrixXd& V_out,
    Eigen::MatrixXd& VN_out,
    Eigen::MatrixXi& F_out)
{
    if (va == vb) return false;
    Eigen::MatrixXi F2 = F;
    for (int i = 0; i < (int)F2.rows(); ++i)
        for (int k = 0; k < 4; ++k)
            if (F2(i,k) == vb) F2(i,k) = va;

    std::vector<std::array<int,4>> kept;
    kept.reserve(F2.rows());
    for (int i = 0; i < (int)F2.rows(); ++i) {
        std::set<int> uniq;
        for (int k = 0; k < 4; ++k) uniq.insert(F2(i,k));
        if ((int)uniq.size() == 4) {
            kept.push_back({F2(i,0), F2(i,1), F2(i,2), F2(i,3)});
        }
    }
    if (kept.empty()) return false;

    V_out = V;
    VN_out = VN;
    V_out.row(va) = 0.5 * (V.row(va) + V.row(vb));
    Eigen::Vector3d n = (VN.row(va) + VN.row(vb)).transpose();
    double nn = n.norm();
    if (nn > 1e-10) VN_out.row(va) = (n/nn).transpose();
    else            VN_out.row(va) = VN.row(va);

    F_out.resize((int)kept.size(), 4);
    for (int i = 0; i < (int)kept.size(); ++i)
        for (int k = 0; k < 4; ++k)
            F_out(i,k) = kept[i][k];
    compress_quad_mesh(V_out, VN_out, F_out);
    return true;
}

static bool merge_vertex_pairs_trial(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& VN,
    const Eigen::MatrixXi& F,
    const std::vector<std::pair<int,int>>& merges,
    Eigen::MatrixXd& V_out,
    Eigen::MatrixXd& VN_out,
    Eigen::MatrixXi& F_out)
{
    if (merges.empty()) return false;
    std::map<int,int> repl;
    std::set<int> lhs, rhs;
    for (const auto& m : merges) {
        int va = m.first;
        int vb = m.second;
        if (va == vb) return false;
        if (lhs.count(va) || lhs.count(vb) || rhs.count(va) || rhs.count(vb)) return false;
        lhs.insert(va);
        rhs.insert(vb);
        repl[vb] = va;
    }

    Eigen::MatrixXi F2 = F;
    for (int i = 0; i < (int)F2.rows(); ++i)
        for (int k = 0; k < 4; ++k) {
            auto it = repl.find(F2(i,k));
            if (it != repl.end()) F2(i,k) = it->second;
        }

    std::vector<std::array<int,4>> kept;
    kept.reserve(F2.rows());
    for (int i = 0; i < (int)F2.rows(); ++i) {
        std::set<int> uniq;
        for (int k = 0; k < 4; ++k) uniq.insert(F2(i,k));
        if ((int)uniq.size() == 4)
            kept.push_back({F2(i,0), F2(i,1), F2(i,2), F2(i,3)});
    }
    if (kept.empty()) return false;

    V_out = V;
    VN_out = VN;
    for (const auto& m : merges) {
        int va = m.first;
        int vb = m.second;
        V_out.row(va) = 0.5 * (V_out.row(va) + V.row(vb));
        Eigen::Vector3d n = (VN_out.row(va) + VN.row(vb)).transpose();
        double nn = n.norm();
        if (nn > 1e-10) VN_out.row(va) = (n / nn).transpose();
    }

    F_out.resize((int)kept.size(), 4);
    for (int i = 0; i < (int)kept.size(); ++i)
        for (int k = 0; k < 4; ++k)
            F_out(i,k) = kept[i][k];
    compress_quad_mesh(V_out, VN_out, F_out);
    return true;
}

static void try_close_single_boundary_slit(
    Eigen::MatrixXd& quadV,
    Eigen::MatrixXd& quadVN,
    Eigen::MatrixXi& quadF)
{
    BoundaryStats before = boundary_stats_quad(quadF);
    if (before.boundary_loops != 1 || before.boundary_chains != 0) return;
    int accepted = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        auto loops = extract_boundary_loops_quad(quadF);
        if (loops.size() != 1) break;
        const std::vector<int>& loop = loops[0];
        if ((int)loop.size() < 64) break;

        Eigen::MatrixXd pts(loop.size(), 3);
        for (int i = 0; i < (int)loop.size(); ++i) pts.row(i) = quadV.row(loop[i]);
        double bbox_diag = (pts.colwise().maxCoeff() - pts.colwise().minCoeff()).norm();
        const int min_gap = std::max(6, (int)std::ceil(0.30 * (double)loop.size()));

        struct PairCand { int i, j; double d; };
        std::vector<PairCand> cands;
        for (int i = 0; i < (int)loop.size(); ++i) {
            double best_d = 1e100;
            int best_j = -1;
            for (int j = 0; j < (int)loop.size(); ++j) {
                if (i == j) continue;
                int cyc = std::min((j - i + (int)loop.size()) % (int)loop.size(),
                                   (i - j + (int)loop.size()) % (int)loop.size());
                if (cyc < min_gap) continue;
                double d = (quadV.row(loop[i]) - quadV.row(loop[j])).norm();
                if (d < best_d) { best_d = d; best_j = j; }
            }
            if (best_j >= 0) cands.push_back({i, best_j, best_d});
        }

        auto try_reciprocal_pairs = [&](const std::vector<PairCand>& in_cands,
                                        const std::vector<double>& dist_ratios,
                                        int min_gap_local,
                                        int max_gap_local) -> bool {
            auto is_reciprocal_under = [&](int a, int b, double max_dist) -> bool {
                for (const auto& r : in_cands) {
                    if (r.i == a && r.j == b && r.d <= max_dist) return true;
                }
                return false;
            };

            for (double ratio : dist_ratios) {
                double max_dist = ratio * std::max(1e-8, bbox_diag);
                for (const auto& c : in_cands) {
                    if (c.i >= c.j) continue;
                    int cyc = std::min((c.j - c.i + (int)loop.size()) % (int)loop.size(),
                                       (c.i - c.j + (int)loop.size()) % (int)loop.size());
                    if (cyc < min_gap_local || cyc > max_gap_local) continue;
                    if (c.d > max_dist) continue;
                    bool reciprocal = is_reciprocal_under(c.j, c.i, max_dist);
                    if (!reciprocal) continue;

                    int n = (int)loop.size();

                    // Before falling back to isolated pair welding, try a short
                    // symmetric strip merge across the slit. This targets the
                    // common case where the seam is a locally offset zipper and
                    // one-vertex welds are too weak to reduce the long loop.
                    std::vector<std::vector<std::pair<int,int>>> strip_merge_sets;
                    {
                        std::vector<std::pair<int,int>> m2a{
                            {loop[c.i], loop[c.j]},
                            {loop[(c.i + 1) % n], loop[(c.j - 1 + n) % n]},
                        };
                        std::vector<std::pair<int,int>> m2b{
                            {loop[c.i], loop[c.j]},
                            {loop[(c.i - 1 + n) % n], loop[(c.j + 1) % n]},
                        };
                        std::vector<std::pair<int,int>> m3{
                            {loop[c.i], loop[c.j]},
                            {loop[(c.i + 1) % n], loop[(c.j - 1 + n) % n]},
                            {loop[(c.i - 1 + n) % n], loop[(c.j + 1) % n]},
                        };
                        strip_merge_sets.push_back(m3);
                        strip_merge_sets.push_back(m2a);
                        strip_merge_sets.push_back(m2b);
                    }

                    for (const auto& merges : strip_merge_sets) {
                        bool valid_strip = true;
                        std::set<int> used_vs;
                        std::set<int> used_vt;
                        for (const auto& m : merges) {
                            int vs = m.first;
                            int vt = m.second;
                            if (vs == vt || used_vs.count(vs) || used_vt.count(vt)) {
                                valid_strip = false;
                                break;
                            }
                            used_vs.insert(vs);
                            used_vt.insert(vt);
                        }
                        if (!valid_strip) continue;

                        bool dist_ok = true;
                        for (const auto& m : merges) {
                            double d = (quadV.row(m.first) - quadV.row(m.second)).norm();
                            if (d > 1.75 * max_dist) {
                                dist_ok = false;
                                break;
                            }
                        }
                        if (!dist_ok) continue;

                        bool reciprocal_strip = true;
                        for (const auto& m : merges) {
                            int ai = -1, bi = -1;
                            for (int t = 0; t < n; ++t) {
                                if (loop[t] == m.first) ai = t;
                                if (loop[t] == m.second) bi = t;
                            }
                            if (ai < 0 || bi < 0 || !is_reciprocal_under(bi, ai, 1.75 * max_dist)) {
                                reciprocal_strip = false;
                                break;
                            }
                        }
                        if (!reciprocal_strip) continue;

                        Eigen::MatrixXd V_trys, VN_trys;
                        Eigen::MatrixXi F_trys;
                        if (!merge_vertex_pairs_trial(quadV, quadVN, quadF, merges, V_trys, VN_trys, F_trys))
                            continue;

                        BoundaryStats afters = boundary_stats_quad(F_trys);
                        if (
                            afters.boundary_edges < before.boundary_edges &&
                            afters.boundary_loops <= before.boundary_loops &&
                            afters.boundary_chains == 0 &&
                            afters.irregular_boundary_vertices <= before.irregular_boundary_vertices
                        ) {
                            quadV = V_trys;
                            quadVN = VN_trys;
                            quadF = F_trys;
                            before = afters;
                            accepted += (int)merges.size();
                            changed = true;
                            return true;
                        }
                    }

                    Eigen::MatrixXd V_try, VN_try;
                    Eigen::MatrixXi F_try;
                    if (!merge_vertex_pair_trial(quadV, quadVN, quadF, loop[c.i], loop[c.j], V_try, VN_try, F_try))
                        continue;

                    BoundaryStats after = boundary_stats_quad(F_try);
                    if (
                        after.boundary_edges < before.boundary_edges &&
                        after.boundary_loops <= before.boundary_loops &&
                        after.boundary_chains == 0 &&
                        after.irregular_boundary_vertices <= before.irregular_boundary_vertices
                    ) {
                        quadV = V_try;
                        quadVN = VN_try;
                        quadF = F_try;
                        before = after;
                        ++accepted;
                        changed = true;
                        return true;
                    }
                    std::vector<std::pair<int,int>> neighbor_pairs = {
                        {loop[(c.i - 1 + n) % n], loop[(c.j + 1) % n]},
                        {loop[(c.i + 1) % n],     loop[(c.j - 1 + n) % n]},
                    };
                    for (const auto& np : neighbor_pairs) {
                        if (np.first == loop[c.i] || np.first == loop[c.j] ||
                            np.second == loop[c.i] || np.second == loop[c.j]) continue;
                        double d2 = (quadV.row(np.first) - quadV.row(np.second)).norm();
                        if (d2 > 1.5 * max_dist) continue;
                        Eigen::MatrixXd V_try2, VN_try2;
                        Eigen::MatrixXi F_try2;
                        std::vector<std::pair<int,int>> merges{
                            {loop[c.i], loop[c.j]},
                            {np.first,  np.second},
                        };
                        if (!merge_vertex_pairs_trial(quadV, quadVN, quadF, merges, V_try2, VN_try2, F_try2))
                            continue;
                        BoundaryStats after2 = boundary_stats_quad(F_try2);
                        if (
                            after2.boundary_edges < before.boundary_edges &&
                            after2.boundary_loops <= before.boundary_loops &&
                            after2.boundary_chains == 0 &&
                            after2.irregular_boundary_vertices <= before.irregular_boundary_vertices
                        ) {
                            quadV = V_try2;
                            quadVN = VN_try2;
                            quadF = F_try2;
                            before = after2;
                            accepted += 2;
                            changed = true;
                            return true;
                        }
                    }
                }
            }
            return false;
        };

        if (try_reciprocal_pairs(cands, {0.0015, 0.002, 0.003, 0.004, 0.006}, 8, 48))
            continue;

        std::vector<double> dist_ratios = {0.004, 0.006, 0.008, 0.010, 0.014, 0.020};
        bool improved = false;
        for (double ratio : dist_ratios) {
            double max_dist = ratio * std::max(1e-8, bbox_diag);
            for (const auto& c : cands) {
                if (c.d > max_dist || c.i >= c.j) continue;
                bool reciprocal = false;
                for (const auto& r : cands) {
                    if (r.i == c.j && r.j == c.i && r.d <= max_dist) { reciprocal = true; break; }
                }
                if (!reciprocal) continue;

                Eigen::MatrixXd V_try, VN_try;
                Eigen::MatrixXi F_try;
                if (!merge_vertex_pair_trial(quadV, quadVN, quadF, loop[c.i], loop[c.j], V_try, VN_try, F_try))
                    continue;

                BoundaryStats after = boundary_stats_quad(F_try);
                if (
                    after.boundary_edges < before.boundary_edges &&
                    after.boundary_loops <= before.boundary_loops &&
                    after.boundary_chains == 0 &&
                    after.irregular_boundary_vertices <= before.irregular_boundary_vertices
                ) {
                    quadV = V_try;
                    quadVN = VN_try;
                    quadF = F_try;
                    before = after;
                    ++accepted;
                    improved = true;
                    changed = true;
                    break;
                }
            }
            if (improved) break;
        }
    }
    if (accepted > 0) {
        std::cout << "[run_miq] Slit weld accepted: "
                  << accepted << " merges -> edges=" << before.boundary_edges
                  << ", loops=" << before.boundary_loops << "\n";
    }
}

static void try_absorb_small_boundary_loop(
    Eigen::MatrixXd& quadV,
    Eigen::MatrixXd& quadVN,
    Eigen::MatrixXi& quadF)
{
    BoundaryStats before = boundary_stats_quad(quadF);
    if (before.boundary_loops != 2 || before.boundary_chains != 0) return;

    auto loops = extract_boundary_loops_quad(quadF);
    if (loops.size() != 2) return;

    int small_idx = (loops[0].size() <= loops[1].size()) ? 0 : 1;
    int large_idx = 1 - small_idx;
    std::vector<int> small = loops[small_idx];
    std::vector<int> large = loops[large_idx];
    if ((int)small.size() < 8 || (int)small.size() > 96 || (int)large.size() < 96) return;

    Eigen::MatrixXd pts(quadV.rows(), 3);
    pts = quadV;
    double bbox_diag = (pts.colwise().maxCoeff() - pts.colwise().minCoeff()).norm();
    if (bbox_diag < 1e-8) return;

    struct PairCand { int vs, vl; double d; };
    auto build_candidates = [&](double max_dist) {
        std::vector<PairCand> cands;
        std::map<int, std::pair<double,int>> s_to_l;
        std::map<int, std::pair<double,int>> l_to_s;
        for (int vs : small) {
            double best_d = 1e100;
            int best_l = -1;
            for (int vl : large) {
                double d = (quadV.row(vs) - quadV.row(vl)).norm();
                if (d < best_d) {
                    best_d = d;
                    best_l = vl;
                }
            }
            if (best_l >= 0) s_to_l[vs] = {best_d, best_l};
        }
        for (int vl : large) {
            double best_d = 1e100;
            int best_s = -1;
            for (int vs : small) {
                double d = (quadV.row(vl) - quadV.row(vs)).norm();
                if (d < best_d) {
                    best_d = d;
                    best_s = vs;
                }
            }
            if (best_s >= 0) l_to_s[vl] = {best_d, best_s};
        }
        for (const auto& kv : s_to_l) {
            int vs = kv.first;
            double d = kv.second.first;
            int vl = kv.second.second;
            auto it = l_to_s.find(vl);
            if (it == l_to_s.end()) continue;
            if (it->second.second != vs) continue;
            if (d > max_dist || it->second.first > max_dist) continue;
            cands.push_back({vs, vl, d});
        }
        std::sort(cands.begin(), cands.end(), [](const PairCand& a, const PairCand& b) {
            return a.d < b.d;
        });
        return cands;
    };

    int accepted = 0;
    for (double ratio : {0.06, 0.08, 0.10, 0.12}) {
        double max_dist = ratio * bbox_diag;
        bool changed = true;
        while (changed) {
            changed = false;
            loops = extract_boundary_loops_quad(quadF);
            if (loops.size() != 2) break;
            small_idx = (loops[0].size() <= loops[1].size()) ? 0 : 1;
            large_idx = 1 - small_idx;
            small = loops[small_idx];
            large = loops[large_idx];
            if ((int)small.size() < 8 || (int)small.size() > 96 || (int)large.size() < 96) break;

            auto cands = build_candidates(max_dist);
            if (cands.empty()) break;
            if (before.boundary_loops == 2) {
                bool local_accepted = false;
                for (const auto& c : cands) {
                    auto it_small = std::find(small.begin(), small.end(), c.vs);
                    auto it_large = std::find(large.begin(), large.end(), c.vl);
                    if (it_small == small.end() || it_large == large.end()) continue;
                    int si = (int)std::distance(small.begin(), it_small);
                    int li = (int)std::distance(large.begin(), it_large);
                    std::vector<std::pair<int,int>> neighbor_pairs = {
                        {large[(li - 1 + (int)large.size()) % (int)large.size()], small[(si + 1) % (int)small.size()]},
                        {large[(li + 1) % (int)large.size()],                 small[(si + 1) % (int)small.size()]},
                        {large[(li - 1 + (int)large.size()) % (int)large.size()], small[(si - 1 + (int)small.size()) % (int)small.size()]},
                        {large[(li + 1) % (int)large.size()],                 small[(si - 1 + (int)small.size()) % (int)small.size()]},
                    };
                    for (const auto& np : neighbor_pairs) {
                        if (np.first == c.vl || np.second == c.vs) continue;
                        double d2 = (quadV.row(np.first) - quadV.row(np.second)).norm();
                        if (d2 > 1.75 * max_dist) continue;
                        Eigen::MatrixXd V_try, VN_try;
                        Eigen::MatrixXi F_try;
                        std::vector<std::pair<int,int>> merges{{c.vl, c.vs}, np};
                        if (!merge_vertex_pairs_trial(quadV, quadVN, quadF, merges, V_try, VN_try, F_try))
                            continue;
                        BoundaryStats after = boundary_stats_quad(F_try);
                        if (
                            after.boundary_edges < before.boundary_edges &&
                            after.boundary_loops <= before.boundary_loops &&
                            after.boundary_chains == 0 &&
                            after.irregular_boundary_vertices <= before.irregular_boundary_vertices
                        ) {
                            quadV = V_try;
                            quadVN = VN_try;
                            quadF = F_try;
                            before = after;
                            accepted += 2;
                            changed = true;
                            local_accepted = true;
                            break;
                        }
                    }
                    if (local_accepted) break;
                }
                if (local_accepted) continue;
            }
            if (before.boundary_loops == 2 && cands.size() >= 2) {
                int combo_limit = std::min<int>(6, cands.size());
                bool combo_accepted = false;
                for (int i = 0; i < combo_limit && !combo_accepted; ++i) {
                    for (int j = i + 1; j < combo_limit && !combo_accepted; ++j) {
                        if (cands[i].vs == cands[j].vs || cands[i].vl == cands[j].vl) continue;
                        Eigen::MatrixXd V_try, VN_try;
                        Eigen::MatrixXi F_try;
                        std::vector<std::pair<int,int>> merges{
                            {cands[i].vl, cands[i].vs},
                            {cands[j].vl, cands[j].vs},
                        };
                        if (!merge_vertex_pairs_trial(quadV, quadVN, quadF, merges, V_try, VN_try, F_try))
                            continue;
                        BoundaryStats after = boundary_stats_quad(F_try);
                        if (
                            after.boundary_edges < before.boundary_edges &&
                            after.boundary_loops <= before.boundary_loops &&
                            after.boundary_chains == 0 &&
                            after.irregular_boundary_vertices <= before.irregular_boundary_vertices
                        ) {
                            quadV = V_try;
                            quadVN = VN_try;
                            quadF = F_try;
                            before = after;
                            accepted += 2;
                            changed = true;
                            combo_accepted = true;
                        }
                    }
                }
                if (combo_accepted) continue;
            }
            for (const auto& c : cands) {
                Eigen::MatrixXd V_try, VN_try;
                Eigen::MatrixXi F_try;
                if (!merge_vertex_pair_trial(quadV, quadVN, quadF, c.vl, c.vs, V_try, VN_try, F_try))
                    continue;
                BoundaryStats after = boundary_stats_quad(F_try);
                if (
                    after.boundary_edges < before.boundary_edges &&
                    after.boundary_loops <= before.boundary_loops &&
                    after.boundary_chains == 0 &&
                    after.irregular_boundary_vertices <= before.irregular_boundary_vertices
                ) {
                    quadV = V_try;
                    quadVN = VN_try;
                    quadF = F_try;
                    before = after;
                    ++accepted;
                    changed = true;
                    break;
                }
            }
        }
        if (before.boundary_loops <= 1) break;
    }
    if (accepted > 0) {
        std::cout << "[run_miq] Absorbed small boundary loop with " << accepted
                  << " welds -> edges=" << before.boundary_edges
                  << ", loops=" << before.boundary_loops << "\n";
    }
}

// ─── Barycentric integer-grid quad extraction ─────────────────────────────────
//
//  For each UV triangle, finds integer grid points (p,q) inside it via bounding-
//  box enumeration, computes their 3D positions by barycentric interpolation, and
//  assembles quad faces from 2×2 cells.
//
//  Key properties vs old nearest-vertex approach:
//  - Creates NEW vertices at exact integer UV positions (no snap distance limit).
//  - UV fold-over triangles (det < 0) are SKIPPED to avoid coverage conflicts.
//  - Seam edges are handled: first-found triangle at a seam grid point wins.

static bool extract_quads_barycentric(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& VN,
    const Eigen::MatrixXd& UV,
    const Eigen::MatrixXi& FUV,
    const Eigen::MatrixXi& F3,
    Eigen::MatrixXd& quadV,
    Eigen::MatrixXd& quadVN,
    Eigen::MatrixXi& quadF)
{
    using Pii = std::pair<int,int>;

    int F_count = (int)FUV.rows();
    std::map<Pii, Eigen::Vector3d> grid_pos;
    std::map<Pii, Eigen::Vector3d> grid_nor;
    std::map<Pii, int> grid_hits;
    std::map<Pii, int> grid_fill_depth;
    int fold_overs = 0;
    int recovered_fold_overs = 0;
    int degenerate_uv = 0;
    int seam_consensus_merges = 0;
    const double mesh_diag = (V.colwise().maxCoeff() - V.colwise().minCoeff()).norm();
    const double seam_merge_dist = 0.01 * std::max(1e-8, mesh_diag);

    for (int f = 0; f < F_count; ++f) {
        Eigen::Vector2d uv0 = UV.row(FUV(f,0)).transpose();
        Eigen::Vector2d uv1 = UV.row(FUV(f,1)).transpose();
        Eigen::Vector2d uv2 = UV.row(FUV(f,2)).transpose();

        Eigen::Vector3d v0 = V.row(F3(f,0)).transpose();
        Eigen::Vector3d v1 = V.row(F3(f,1)).transpose();
        Eigen::Vector3d v2 = V.row(F3(f,2)).transpose();
        Eigen::Vector3d n0 = VN.row(F3(f,0)).transpose();
        Eigen::Vector3d n1 = VN.row(F3(f,1)).transpose();
        Eigen::Vector3d n2 = VN.row(F3(f,2)).transpose();

        Eigen::Matrix2d T;
        T.col(0) = uv1 - uv0;
        T.col(1) = uv2 - uv0;
        double det = T.determinant();

        // Degenerate UV triangles still cannot contribute usable grid cells.
        // For negative-oriented (fold-over) triangles, do NOT discard the whole
        // region immediately: reorder the local UV/3D correspondence to restore
        // positive orientation, then continue barycentric sampling. This reduces
        // hole creation from locally inverted parametrisation patches.
        if (std::abs(det) < 1e-10) { ++degenerate_uv; continue; }
        if (det < 0.0) {
            std::swap(uv1, uv2);
            std::swap(v1,  v2);
            std::swap(n1,  n2);
            T.col(0) = uv1 - uv0;
            T.col(1) = uv2 - uv0;
            det = T.determinant();
            if (det < 1e-10) { ++fold_overs; continue; }
            ++recovered_fold_overs;
        }

        Eigen::Matrix2d T_inv = T.inverse();

        const double eps = 1e-8;
        double u_lo = std::min({uv0[0],uv1[0],uv2[0]});
        double u_hi = std::max({uv0[0],uv1[0],uv2[0]});
        double v_lo = std::min({uv0[1],uv1[1],uv2[1]});
        double v_hi = std::max({uv0[1],uv1[1],uv2[1]});

        int pi0=(int)std::ceil (u_lo-eps), pi1=(int)std::floor(u_hi+eps);
        int qi0=(int)std::ceil (v_lo-eps), qi1=(int)std::floor(v_hi+eps);

        for (int pi = pi0; pi <= pi1; ++pi) {
            for (int qi = qi0; qi <= qi1; ++qi) {
                Eigen::Vector2d p(pi, qi);
                Eigen::Vector2d lam = T_inv * (p - uv0);
                double w1 = lam[0], w2 = lam[1], w0 = 1.0 - w1 - w2;

                const double tol = -1e-7;
                if (w0 < tol || w1 < tol || w2 < tol) continue;

                w0=std::max(0.0,w0); w1=std::max(0.0,w1); w2=std::max(0.0,w2);
                double ws = w0+w1+w2;
                if (ws < 1e-14) continue;
                w0/=ws; w1/=ws; w2/=ws;

                Pii key = {pi, qi};
                Eigen::Vector3d pos = w0*v0 + w1*v1 + w2*v2;
                Eigen::Vector3d n = w0*n0 + w1*n1 + w2*n2;
                double nn = n.norm();
                Eigen::Vector3d nor = (nn > 1e-10) ? (n/nn).eval() : n0;

                auto it = grid_pos.find(key);
                if (it != grid_pos.end()) {
                    if ((it->second - pos).norm() <= seam_merge_dist) {
                        int cnt = std::max(1, grid_hits[key]);
                        double a = 1.0 / double(cnt + 1);
                        it->second = (1.0 - a) * it->second + a * pos;
                        Eigen::Vector3d n_mix = (1.0 - a) * grid_nor[key] + a * nor;
                        double n_mix_n = n_mix.norm();
                        grid_nor[key] = (n_mix_n > 1e-10) ? (n_mix / n_mix_n).eval() : grid_nor[key];
                        grid_hits[key] = cnt + 1;
                        ++seam_consensus_merges;
                    }
                    continue;
                }

                grid_pos[key] = pos;
                grid_nor[key] = nor;
                grid_hits[key] = 1;
                grid_fill_depth[key] = 0;
            }
        }
    }

    std::cout << "[run_miq] UV fold-overs recovered: " << recovered_fold_overs
              << ", skipped: " << fold_overs
              << ", degenerate: " << degenerate_uv
              << " / " << F_count << " UV triangles\n";
    if (seam_consensus_merges > 0)
        std::cout << "[run_miq] Seam-consensus merged duplicate grid hits: "
                  << seam_consensus_merges << "\n";
    if (grid_pos.empty()) return false;

    std::map<Pii,int> grid_to_idx;
    int p_min=INT_MAX, p_max=INT_MIN, q_min=INT_MAX, q_max=INT_MIN;
    for (auto& kv : grid_pos) {
        p_min=std::min(p_min,kv.first.first);  p_max=std::max(p_max,kv.first.first);
        q_min=std::min(q_min,kv.first.second); q_max=std::max(q_max,kv.first.second);
    }

    int crack_fills = 0;
    int strip_fills = 0;
    int cell_corner_fills = 0;
    const int max_fill_passes = 8;
    for (int pass = 0; pass < max_fill_passes; ++pass) {
        int added_this_pass = 0;

        {
            std::vector<std::tuple<Pii, Eigen::Vector3d, Eigen::Vector3d>> staged;
            for (int p = p_min + 1; p < p_max; ++p) {
                for (int q = q_min + 1; q < q_max; ++q) {
                    Pii key = {p, q};
                    if (grid_pos.count(key)) continue;

                    const Pii left  = {p - 1, q};
                    const Pii right = {p + 1, q};
                    const Pii down  = {p, q - 1};
                    const Pii up    = {p, q + 1};
                    if (!grid_pos.count(left) || !grid_pos.count(right) ||
                        !grid_pos.count(down) || !grid_pos.count(up))
                        continue;
                    if (grid_fill_depth[left] > 1 || grid_fill_depth[right] > 1 ||
                        grid_fill_depth[down] > 1 || grid_fill_depth[up] > 1)
                        continue;

                    Eigen::Vector3d pos = 0.25 * (
                        grid_pos[left] + grid_pos[right] + grid_pos[down] + grid_pos[up]
                    );
                    Eigen::Vector3d nor = 0.25 * (
                        grid_nor[left] + grid_nor[right] + grid_nor[down] + grid_nor[up]
                    );
                    double nn = nor.norm();
                    if (nn > 1e-10) nor /= nn;
                    staged.push_back({key, pos, nor});
                }
            }
            for (const auto& item : staged) {
                const Pii& key = std::get<0>(item);
                if (grid_pos.count(key)) continue;
                grid_pos[key] = std::get<1>(item);
                grid_nor[key] = std::get<2>(item);
                grid_hits[key] = 1;
                grid_fill_depth[key] = pass + 1;
                ++crack_fills;
                ++added_this_pass;
            }
        }

        {
            std::vector<std::tuple<Pii, Eigen::Vector3d, Eigen::Vector3d>> staged;
            for (int p = p_min + 1; p < p_max - 1; ++p) {
                for (int q = q_min; q < q_max; ++q) {
                    const Pii km = {p - 1, q};
                    const Pii k  = {p, q};
                    const Pii kp = {p + 1, q};
                    if (grid_pos.count(k) || !grid_pos.count(km) || !grid_pos.count(kp))
                        continue;
                    if (grid_fill_depth[km] > 0 || grid_fill_depth[kp] > 0)
                        continue;
                    Eigen::Vector3d pos = 0.5 * (grid_pos[km] + grid_pos[kp]);
                    Eigen::Vector3d nor = 0.5 * (grid_nor[km] + grid_nor[kp]);
                    double nn = nor.norm();
                    if (nn > 1e-10) nor /= nn;
                    staged.push_back({k, pos, nor});
                }
            }
            for (int p = p_min; p < p_max; ++p) {
                for (int q = q_min + 1; q < q_max - 1; ++q) {
                    const Pii km = {p, q - 1};
                    const Pii k  = {p, q};
                    const Pii kp = {p, q + 1};
                    if (grid_pos.count(k) || !grid_pos.count(km) || !grid_pos.count(kp))
                        continue;
                    if (grid_fill_depth[km] > 0 || grid_fill_depth[kp] > 0)
                        continue;
                    Eigen::Vector3d pos = 0.5 * (grid_pos[km] + grid_pos[kp]);
                    Eigen::Vector3d nor = 0.5 * (grid_nor[km] + grid_nor[kp]);
                    double nn = nor.norm();
                    if (nn > 1e-10) nor /= nn;
                    staged.push_back({k, pos, nor});
                }
            }
            for (const auto& item : staged) {
                const Pii& key = std::get<0>(item);
                if (grid_pos.count(key)) continue;
                grid_pos[key] = std::get<1>(item);
                grid_nor[key] = std::get<2>(item);
                grid_hits[key] = 1;
                grid_fill_depth[key] = pass + 1;
                ++strip_fills;
                ++added_this_pass;
            }
        }

        {
            std::vector<std::tuple<Pii, Eigen::Vector3d, Eigen::Vector3d>> staged;
            for (int p = p_min; p < p_max; ++p) {
                for (int q = q_min; q < q_max; ++q) {
                    const Pii k00 = {p,   q};
                    const Pii k10 = {p+1, q};
                    const Pii k11 = {p+1, q+1};
                    const Pii k01 = {p,   q+1};

                    const bool h00 = grid_pos.count(k00);
                    const bool h10 = grid_pos.count(k10);
                    const bool h11 = grid_pos.count(k11);
                    const bool h01 = grid_pos.count(k01);
                    const int num = int(h00) + int(h10) + int(h11) + int(h01);
                    if (num != 3) continue;
                    int seed_supports = int(h00 && grid_fill_depth[k00] == 0)
                                      + int(h10 && grid_fill_depth[k10] == 0)
                                      + int(h11 && grid_fill_depth[k11] == 0)
                                      + int(h01 && grid_fill_depth[k01] == 0);
                    int max_support_depth = 0;
                    if (h00) max_support_depth = std::max(max_support_depth, grid_fill_depth[k00]);
                    if (h10) max_support_depth = std::max(max_support_depth, grid_fill_depth[k10]);
                    if (h11) max_support_depth = std::max(max_support_depth, grid_fill_depth[k11]);
                    if (h01) max_support_depth = std::max(max_support_depth, grid_fill_depth[k01]);
                    if (seed_supports < 2 || max_support_depth > 1) continue;

                    Pii miss;
                    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
                    Eigen::Vector3d nor = Eigen::Vector3d::Zero();
                    if (!h00) {
                        miss = k00;
                        pos = grid_pos[k10] + grid_pos[k01] - grid_pos[k11];
                        nor = grid_nor[k10] + grid_nor[k01] - grid_nor[k11];
                    } else if (!h10) {
                        miss = k10;
                        pos = grid_pos[k00] + grid_pos[k11] - grid_pos[k01];
                        nor = grid_nor[k00] + grid_nor[k11] - grid_nor[k01];
                    } else if (!h11) {
                        miss = k11;
                        pos = grid_pos[k10] + grid_pos[k01] - grid_pos[k00];
                        nor = grid_nor[k10] + grid_nor[k01] - grid_nor[k00];
                    } else {
                        miss = k01;
                        pos = grid_pos[k00] + grid_pos[k11] - grid_pos[k10];
                        nor = grid_nor[k00] + grid_nor[k11] - grid_nor[k10];
                    }

                    double nn = nor.norm();
                    if (nn > 1e-10) nor /= nn;
                    staged.push_back({miss, pos, nor});
                }
            }
            for (const auto& item : staged) {
                const Pii& key = std::get<0>(item);
                if (grid_pos.count(key)) continue;
                grid_pos[key] = std::get<1>(item);
                grid_nor[key] = std::get<2>(item);
                grid_hits[key] = 1;
                grid_fill_depth[key] = pass + 1;
                ++cell_corner_fills;
                ++added_this_pass;
            }
        }

        if (added_this_pass == 0) break;
    }
    if (crack_fills > 0)
        std::cout << "[run_miq] Grid crack-filled vertices: " << crack_fills << "\n";
    if (strip_fills > 0)
        std::cout << "[run_miq] Grid strip-filled vertices: " << strip_fills << "\n";
    if (cell_corner_fills > 0)
        std::cout << "[run_miq] Cell-corner filled vertices: " << cell_corner_fills << "\n";

    std::cout << "[run_miq] Barycentric grid points: " << grid_pos.size() << "\n";

    std::vector<Eigen::Vector3d> out_verts, out_norms;
    for (auto& kv : grid_pos) {
        grid_to_idx[kv.first] = (int)out_verts.size();
        out_verts.push_back(kv.second);
        out_norms.push_back(grid_nor[kv.first]);
    }

    std::vector<std::array<int,4>> quads_raw;
    for (int p = p_min; p < p_max; ++p)
        for (int q = q_min; q < q_max; ++q) {
            auto f00=grid_to_idx.find({p,  q  }); if(f00==grid_to_idx.end()) continue;
            auto f10=grid_to_idx.find({p+1,q  }); if(f10==grid_to_idx.end()) continue;
            auto f11=grid_to_idx.find({p+1,q+1}); if(f11==grid_to_idx.end()) continue;
            auto f01=grid_to_idx.find({p,  q+1}); if(f01==grid_to_idx.end()) continue;
            quads_raw.push_back({f00->second,f10->second,f11->second,f01->second});
        }

    if (quads_raw.empty()) return false;

    quadV.resize((int)out_verts.size(), 3);
    quadVN.resize((int)out_norms.size(), 3);
    for (int i=0; i<(int)out_verts.size(); ++i) {
        quadV.row(i)  = out_verts[i].transpose();
        quadVN.row(i) = out_norms[i].transpose();
    }
    quadF.resize((int)quads_raw.size(), 4);
    for (int i=0; i<(int)quads_raw.size(); ++i)
        for (int j=0; j<4; ++j)
            quadF(i,j) = quads_raw[i][j];
    return true;
}

// ─── libQEx quad extraction (optional, HAS_LIBQEX) ────────────────────────────
//
//  Uses libQEx's exact-arithmetic transition-function approach to extract a
//  valid quad mesh from MIQ UV output.  Key advantages over barycentric:
//    - Handles fold-overs correctly (skips only truly degenerate faces, not
//      rounded-off ones that barycentric discards).
//    - Produces a proper 2-manifold quad mesh without holes or overlaps.
//    - Uses SSE exact arithmetic for geometric predicates (configured via
//      -msse -mfpmath=sse compiler flag).
//
//  Input format:
//    UV   (nUV × 2): MIQ UV coordinates (one row per UV vertex)
//    FUV  (nF  × 3): per-face UV indices (half-edge UV table from MIQ)
//
//  Returns true on success; quadV / quadF are populated.
//  Returns false if libQEx produces 0 quads (caller falls back to barycentric).

#ifdef HAS_LIBQEX
static bool extract_quads_libqex(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& VN,
    const Eigen::MatrixXd& UV,
    const Eigen::MatrixXi& FUV,
    const Eigen::MatrixXi& F3,
    Eigen::MatrixXd& quadV,
    Eigen::MatrixXd& quadVN,
    Eigen::MatrixXi& quadF)
{
    int nV = (int)V.rows();
    int nF = (int)F3.rows();

    // ── Build qex_TriMesh ───────────────────────────────────────────────────
    std::vector<qex_Point3> verts(nV);
    for (int i = 0; i < nV; ++i) {
        verts[i].x[0] = V(i, 0);
        verts[i].x[1] = V(i, 1);
        verts[i].x[2] = V(i, 2);
    }

    std::vector<qex_Tri> tris(nF);
    for (int f = 0; f < nF; ++f) {
        tris[f].idx[0] = (unsigned int)F3(f, 0);
        tris[f].idx[1] = (unsigned int)F3(f, 1);
        tris[f].idx[2] = (unsigned int)F3(f, 2);
    }

    // Per-triangle UV coordinates (half-edge UV from MIQ's FUV table)
    std::vector<qex_UVTri> uvTris(nF);
    for (int f = 0; f < nF; ++f) {
        for (int k = 0; k < 3; ++k) {
            int ui = FUV(f, k);
            uvTris[f].uvs[k].x[0] = UV(ui, 0);
            uvTris[f].uvs[k].x[1] = UV(ui, 1);
        }
    }

    qex_TriMesh triMesh;
    triMesh.vertex_count = (unsigned int)nV;
    triMesh.tri_count    = (unsigned int)nF;
    triMesh.vertices     = verts.data();
    triMesh.tris         = tris.data();
    triMesh.uvTris       = uvTris.data();

    // ── Call libQEx ─────────────────────────────────────────────────────────
    qex_QuadMesh outMesh;
    outMesh.vertex_count = 0;
    outMesh.quad_count   = 0;
    outMesh.vertices     = nullptr;
    outMesh.quads        = nullptr;

    qex_extractQuadMesh(&triMesh, nullptr, &outMesh);

    std::cout << "[libQEx] Extracted " << outMesh.quad_count << " quads, "
              << outMesh.vertex_count << " vertices.\n";

    if (outMesh.quad_count == 0 || outMesh.vertices == nullptr) {
        if (outMesh.vertices) std::free(outMesh.vertices);
        if (outMesh.quads)    std::free(outMesh.quads);
        return false;
    }

    // ── Convert to Eigen ────────────────────────────────────────────────────
    int nqV = (int)outMesh.vertex_count;
    int nqF = (int)outMesh.quad_count;

    quadV.resize(nqV, 3);
    for (int i = 0; i < nqV; ++i) {
        quadV(i, 0) = outMesh.vertices[i].x[0];
        quadV(i, 1) = outMesh.vertices[i].x[1];
        quadV(i, 2) = outMesh.vertices[i].x[2];
    }

    quadF.resize(nqF, 4);
    for (int i = 0; i < nqF; ++i) {
        for (int j = 0; j < 4; ++j)
            quadF(i, j) = (int)outMesh.quads[i].idx[j];
    }

    std::free(outMesh.vertices);
    std::free(outMesh.quads);

    // ── Compute per-vertex normals from quad faces ───────────────────────────
    // libQEx creates vertices at integer grid positions; compute normals from
    // the extracted quad geometry rather than interpolating from the input mesh.
    quadVN.resize(nqV, 3);
    quadVN.setZero();
    for (int i = 0; i < nqF; ++i) {
        Eigen::Vector3d v0 = quadV.row(quadF(i, 0));
        Eigen::Vector3d v1 = quadV.row(quadF(i, 1));
        Eigen::Vector3d v2 = quadV.row(quadF(i, 2));
        Eigen::Vector3d v3 = quadV.row(quadF(i, 3));
        // Face normal from diagonals (robust for quads)
        Eigen::Vector3d n = (v2 - v0).cross(v3 - v1);
        double nn = n.norm();
        if (nn > 1e-10) n /= nn;
        quadVN.row(quadF(i, 0)) += n;
        quadVN.row(quadF(i, 1)) += n;
        quadVN.row(quadF(i, 2)) += n;
        quadVN.row(quadF(i, 3)) += n;
    }
    for (int i = 0; i < nqV; ++i) {
        double nn = quadVN.row(i).norm();
        if (nn > 1e-10) quadVN.row(i) /= nn;
    }

    return true;
}
#endif  // HAS_LIBQEX


// ─── OBJ writer ───────────────────────────────────────────────────────────────

static void write_quad_obj(const std::string& path,
                           const Eigen::MatrixXd& V,
                           const Eigen::MatrixXi& F)
{
    std::ofstream out(path);
    out << std::fixed; out.precision(8);
    for (int i=0; i<(int)V.rows(); ++i)
        out << "v " << V(i,0) << " " << V(i,1) << " " << V(i,2) << "\n";
    for (int i=0; i<(int)F.rows(); ++i) {
        out << "f";
        for (int j=0; j<(int)F.cols(); ++j)
            out << " " << (F(i,j)+1);
        out << "\n";
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr
            << "Usage: run_miq <mesh.obj> <u_real.txt> <u_imag.txt> <out_quad.obj>\n"
            << "               [gradient_size=20] [stiffness=5]\n"
            << "               [direct_round=1] [iter=5]\n"
            << "               [pd1.txt] [pd2.txt]\n"
            << "\n"
            << "  u_real/u_imag: per-vertex GL cross-field (N×1 each).\n"
            << "  pd1/pd2:       optional per-face directions (F×3); used only\n"
            << "                 for comb_frame_field. Omit for isotropic field.\n";
        return 1;
    }

    const std::string mesh_path   = argv[1];
    const std::string ureal_path  = argv[2];
    const std::string uimag_path  = argv[3];
    const std::string out_path    = argv[4];
    const double gradient_size    = (argc > 5) ? std::stod(argv[5]) : 20.0;
    const double stiffness        = (argc > 6) ? std::stod(argv[6]) :  5.0;
    const bool   direct_round     = (argc > 7) ? (std::stoi(argv[7]) != 0) : true;
    const int    miq_iter         = (argc > 8) ? std::stoi(argv[8])        :  5;
    const std::string pd1_path    = (argc > 9)  ? argv[9]  : "";
    const std::string pd2_path    = (argc > 10) ? argv[10] : "";

    // ── Read triangle mesh ────────────────────────────────────────────────────
    Eigen::MatrixXd V; Eigen::MatrixXi F;
    if (!igl::readOBJ(mesh_path, V, F)) {
        std::cerr << "[run_miq] Failed to read mesh: " << mesh_path << "\n"; return 1;
    }
    if (F.cols() != 3) {
        std::cerr << "[run_miq] Input must be a triangle mesh.\n"; return 1;
    }
    std::cout << "[run_miq] Mesh: " << V.rows() << " vertices, " << F.rows() << " faces\n";
    Eigen::MatrixXd VN = compute_vertex_normals(V, F);

    // ── Read per-vertex GL cross-field u ──────────────────────────────────────
    Eigen::VectorXd u_real, u_imag;
    if (!read_vector_txt(ureal_path, u_real) || !read_vector_txt(uimag_path, u_imag)) {
        std::cerr << "[run_miq] Failed to read u field files.\n"; return 1;
    }
    if (u_real.size() != V.rows() || u_imag.size() != V.rows()) {
        std::cerr << "[run_miq] u field size mismatch: got " << u_real.size()
                  << " expected " << V.rows() << "\n"; return 1;
    }
    std::cout << "[run_miq] GL cross-field: " << u_real.size() << " vertices\n";

    // ── Build per-vertex Duff tangent frames ──────────────────────────────────
    Eigen::MatrixXd E1, E2;
    compute_duff_frames(VN, E1, E2);

    // ── Per-face directions via complex averaging ──────────────────────────────
    //
    //  Convert per-vertex u → per-face PD1/PD2 using complex-space averaging
    //  in each face's tangent plane (exp(4i·alpha) average).  This is the correct
    //  approach for 4-RoSy fields: no branch disambiguation needed, and the result
    //  is consistent with igl::cross_field_mismatch's transport convention.
    std::cout << "[run_miq] Computing per-face directions (complex averaging)…\n";
    Eigen::MatrixXd PD1, PD2;
    compute_face_directions_from_u(V, F, u_real, u_imag, E1, E2, PD1, PD2);

    // If external PD files are provided, use them for ANISOTROPY in the comb
    // step only (override PD1/PD2 for combing, but mismatches always come from u).
    bool use_external_pd = (!pd1_path.empty() && !pd2_path.empty());
    Eigen::MatrixXd PD1_comb = PD1, PD2_comb = PD2;  // for comb step
    if (use_external_pd) {
        Eigen::MatrixXd ePD1, ePD2;
        if (read_matrix_txt(pd1_path, ePD1) && read_matrix_txt(pd2_path, ePD2) &&
            ePD1.rows() == F.rows() && ePD2.rows() == F.rows()) {
            PD1_comb = ePD1; PD2_comb = ePD2;
            std::cout << "[run_miq] Using external PD files for anisotropic comb.\n";
        } else {
            std::cerr << "[run_miq] External PD files invalid; using u-derived PD.\n";
            use_external_pd = false;
        }
    }

    // ── Mismatches via igl (on uncombed field) ───────────────────────────────
    //
    //  Compute mismatches from the u-derived per-face field (before combing).
    //  is_combed=false: igl handles the 4-fold ambiguity by finding minimum mismatch
    //  across all 4 orientations of each face — this is the standard igl pipeline.
    std::cout << "[run_miq] Computing mismatches via igl::cross_field_mismatch…\n";
    Eigen::MatrixXi MMatch;
    igl::cross_field_mismatch(V, F, PD1, PD2, false, MMatch);

    // ── Singularities + Poincaré–Hopf check ─────────────────────────────────
    Eigen::Matrix<int,Eigen::Dynamic,1> isSingular, singularityIndex;
    igl::find_cross_field_singularities(V, F, MMatch, isSingular, singularityIndex);
    std::cout << "[run_miq] Singularities: " << isSingular.sum() << " vertices\n";
    check_poincare_hopf(F, (int)V.rows(), singularityIndex);

    // ── Cut mesh from singularities ───────────────────────────────────────────
    Eigen::MatrixXi seamsF;
    igl::cut_mesh_from_singularities(V, F, MMatch, seamsF);

    // ── Comb frame field ──────────────────────────────────────────────────────
    Eigen::MatrixXd BIS1, BIS2, BIS1c, BIS2c;
    igl::compute_frame_field_bisectors(V, F, PD1_comb, PD2_comb, BIS1, BIS2);
    igl::comb_cross_field(V, F, BIS1, BIS2, BIS1c, BIS2c);

    Eigen::MatrixXd PD1c, PD2c;
    igl::comb_frame_field(V, F, PD1_comb, PD2_comb, BIS1c, BIS2c, PD1c, PD2c);

    // ── Run MIQ ───────────────────────────────────────────────────────────────
    std::cout << "[run_miq] Running MIQ (gradient_size=" << gradient_size
              << ", stiffness=" << stiffness
              << ", direct_round=" << (direct_round ? "true" : "false") << ") …\n";

    Eigen::MatrixXd UV; Eigen::MatrixXi FUV;
    // localIter: local optimization passes after each rounding step.
    // Increased from 5 to 20 to allow more time to resolve fold-overs,
    // especially for meshes with many singularities from holonomy effects.
    igl::copyleft::comiso::miq(
        V, F, PD1c, PD2c,
        MMatch, isSingular, seamsF,
        UV, FUV,
        gradient_size, stiffness,
        direct_round, miq_iter,
        /*localIter=*/20, /*doRound=*/true, /*singularityRound=*/true
    );
    std::cout << "[run_miq] UV: " << UV.rows() << " UV vertices, "
              << FUV.rows() << " UV faces\n";

    // Diagnostic: UV bounding box and area statistics
    {
        double u_min=UV.col(0).minCoeff(), u_max=UV.col(0).maxCoeff();
        double v_min=UV.col(1).minCoeff(), v_max=UV.col(1).maxCoeff();
        std::cout << "[run_miq] UV bbox: [" << u_min << ", " << u_max << "] x ["
                  << v_min << ", " << v_max << "]\n";
        // Compute UV area statistics
        double total_uv_area = 0.0;
        int pos_area = 0, neg_area = 0;
        for (int f = 0; f < (int)FUV.rows(); ++f) {
            Eigen::Vector2d u0=UV.row(FUV(f,0)).transpose();
            Eigen::Vector2d u1=UV.row(FUV(f,1)).transpose();
            Eigen::Vector2d u2=UV.row(FUV(f,2)).transpose();
            Eigen::Matrix2d T; T.col(0)=u1-u0; T.col(1)=u2-u0;
            double det = T.determinant();
            if (det > 1e-10) { total_uv_area += det*0.5; ++pos_area; }
            else if (det < -1e-10) --neg_area;
        }
        std::cout << "[run_miq] UV valid area: " << total_uv_area << " (pos=" << pos_area
                  << " neg=" << -neg_area << ")\n";
        std::cout << "[run_miq] Expected grid points from area: ~"
                  << (int)(total_uv_area) << "\n";
    }

    // ── Quad extraction ───────────────────────────────────────────────────────
    //
    //  Strategy:
    //    1. If built with HAS_LIBQEX: try libQEx (exact arithmetic, manifold).
    //    2. Fall back to barycentric UV sampling (always available).
    //
    //  libQEx is preferred when available: it uses transition functions from the
    //  MIQ parametrisation (the integer jumps at seam edges) to guarantee a
    //  topologically valid, hole-free quad mesh.  The barycentric method may
    //  miss grid points in sparse UV regions and skip fold-over faces entirely.
    Eigen::MatrixXd quadV, quadVN; Eigen::MatrixXi quadF;
    bool extracted = false;

#ifdef HAS_LIBQEX
    std::cout << "[run_miq] Attempting libQEx robust quad extraction…\n";
    extracted = extract_quads_libqex(V, VN, UV, FUV, F, quadV, quadVN, quadF);
    if (!extracted)
        std::cerr << "[run_miq] libQEx returned 0 quads — falling back to barycentric.\n";
#endif

    if (!extracted) {
        std::cout << "[run_miq] Using barycentric UV sampling for quad extraction…\n";
        extracted = extract_quads_barycentric(V, VN, UV, FUV, F, quadV, quadVN, quadF);
    }

    if (!extracted) {
        std::cerr << "[run_miq] Extraction failed. Try larger gradient_size.\n";
        return 1;
    }
    std::cout << "[run_miq] Extracted " << quadF.rows() << " quads, "
              << quadV.rows() << " vertices.\n";

    fix_quad_winding(quadV, quadVN, quadF);
    try_close_single_boundary_slit(quadV, quadVN, quadF);
    try_absorb_small_boundary_loop(quadV, quadVN, quadF);
    try_close_single_boundary_slit(quadV, quadVN, quadF);
    fix_quad_winding(quadV, quadVN, quadF);
    write_quad_obj(out_path, quadV, quadF);
    std::cout << "[run_miq] Written → " << out_path << "\n";
    return 0;
}
