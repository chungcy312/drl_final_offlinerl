/**
 * train_afterstate_ntuple.cpp  ?? Afterstate TD / OTD / OTD+TC trainer for 2584
 *
 * Key improvements over original:
 *  1. Pure TD target: r_{t+1} + V(s'_{t+1})  (γ = 1, no reward_scale / empty_w / adj_w)
 *  2. Optimistic initialisation (v_init):  weights default to v_init / num_tuples
 *  3. Alpha decay schedule: α ??α*0.1 at 50 %, α*0.01 at 75 %
 *  4. Temporal Coherence (TC) learning  (--use-tc flag)
 *  5. New binary format NTUPLE2 that stores the default weight value
 *     (backward-compatible loader: also reads the old NTUPLE1 format)
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static const size_t TUPLE_SIZE = 6;

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------
struct Args {
    int    episodes              = 50000;
    double alpha                 = 0.1;      // initial learning rate (OTD paper: 0.1)
    double alpha_mid             = 0.02;     // alpha after 50 % of episodes
    double alpha_final           = 0.01;    // alpha after 75 % of episodes
    double v_init                = 0.0;      // optimistic init value (try 160000-320000)
    bool   use_tc                = false;    // temporal coherence learning
    std::string save_path        = "n_tuple_weights.fbin";
    int    save_every            = 1000;
    std::string checkpoint_dir   = "checkpoints";
    int    log_every             = 200;
    int    seed                  = 0;
    bool   resume                = false;
    // optional training-time MCTS
    bool   train_use_mcts        = true;
    int    mcts_sims             = 24;
    double mcts_c_puct           = 0.90;
    int    mcts_rollout_depth    = 2;
    int    mcts_trigger_empty_le = 5;
    // dataset collection
    std::string dataset_dir      = "generate_2584_dataset/data";
    int    collect_every         = 1;
    int    stage_episodes        = 1000;
    int    score_bin_width       = 500;
    int    max_per_stage_bin     = 25;
    bool   save_rejected_meta    = false;
};

static bool has_suffix(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// ---------------------------------------------------------------------------
// Progress bar
// ---------------------------------------------------------------------------
struct ProgressBar {
    int total;
    std::chrono::steady_clock::time_point start;
    explicit ProgressBar(int t) : total(t), start(std::chrono::steady_clock::now()) {}

    void update(int cur) {
        double ratio  = total > 0 ? static_cast<double>(cur) / total : 1.0;
        int    width  = 28;
        int    fill   = static_cast<int>(ratio * width);
        auto   now    = std::chrono::steady_clock::now();
        double elapsed= std::chrono::duration<double>(now - start).count();
        double eta    = cur > 0 ? elapsed * (static_cast<double>(total - cur) / cur) : 0.0;

        auto fmt_time = [](double s) -> std::string {
            int h = static_cast<int>(s) / 3600;
            int m = (static_cast<int>(s) % 3600) / 60;
            int ss= static_cast<int>(s) % 60;
            char buf[32]; std::snprintf(buf, sizeof(buf), "%d:%02d:%02d", h, m, ss);
            return buf;
        };

        std::cout << "\r[";
        for (int i = 0; i < width; ++i) std::cout << (i < fill ? '=' : ' ');
        std::cout << "] " << std::setw(3) << static_cast<int>(ratio * 100.0) << "% "
                  << cur << "/" << total
                  << " | elapsed " << fmt_time(elapsed)
                  << " | eta "     << fmt_time(eta)
                  << std::flush;
        if (cur == total) std::cout << "\n";
    }
};

// ---------------------------------------------------------------------------
// Fibonacci helper (2584 tile encoding)
// ---------------------------------------------------------------------------
class FibHelper {
public:
    FibHelper() {
        fib_values = {1, 2};
        while (fib_values.back() < 6000000)
            fib_values.push_back(fib_values[fib_values.size()-1] + fib_values[fib_values.size()-2]);
        fib_code[0] = 0;
        for (size_t i = 0; i < fib_values.size(); ++i)
            fib_code[fib_values[i]] = static_cast<int>(i + 1);
    }
    bool mergeable_value(int a, int b) const {
        if (a <= 0 || b <= 0) return false;
        if (a == 1 && b == 1) return true;
        auto ia = fib_code.find(a), ib = fib_code.find(b);
        if (ia == fib_code.end() || ib == fib_code.end()) return false;
        return std::abs(ia->second - ib->second) == 1;
    }
    bool mergeable_code(int a, int b) const {
        if (a <= 0 || b <= 0) return false;
        if (a == 1 && b == 1) return true;
        return std::abs(a - b) == 1;
    }
    int value_to_code(int v) const {
        auto it = fib_code.find(v);
        return it == fib_code.end() ? 0 : it->second;
    }
private:
    std::vector<int> fib_values;
    std::unordered_map<int,int> fib_code;
};
static const FibHelper FIB;

// ---------------------------------------------------------------------------
// Board types and helpers
// ---------------------------------------------------------------------------
using Board = std::array<int,16>;
using Codes = std::array<int,16>;

struct MoveResult { Board board{}; int reward = 0; bool moved = false; };

static inline int idx(int r, int c) { return r*4+c; }

Codes board_to_codes(const Board& b) {
    Codes c{};
    for (int i = 0; i < 16; ++i) c[i] = FIB.value_to_code(b[i]);
    return c;
}
int count_empty(const Codes& c) {
    int n=0; for (int x : c) n += (x==0); return n;
}

std::pair<std::array<int,4>, int> merge_line_left(const std::array<int,4>& line) {
    std::vector<int> nz;
    nz.reserve(4);
    for (int x : line) if (x) nz.push_back(x);
    std::array<int,4> out{0,0,0,0};
    int top=0, reward=0; size_t i=0;
    while (i < nz.size()) {
        if (i+1 < nz.size() && FIB.mergeable_value(nz[i], nz[i+1])) {
            out[top++] = nz[i]+nz[i+1]; reward += nz[i]+nz[i+1]; i+=2;
        } else { out[top++] = nz[i++]; }
    }
    return {out, reward};
}

MoveResult apply_action(const Board& b, int action) {
    Board out{};
    int reward=0;
    if (action==0) {
        for (int c=0;c<4;++c) {
            std::array<int,4> line{b[idx(0,c)],b[idx(1,c)],b[idx(2,c)],b[idx(3,c)]};
            auto [m,r]=merge_line_left(line);
            for (int rr=0;rr<4;++rr) {
                out[idx(rr,c)] = m[rr];
            }
            reward += r;
        }
    } else if (action==1) {
        for (int c=0;c<4;++c) {
            std::array<int,4> line{b[idx(3,c)],b[idx(2,c)],b[idx(1,c)],b[idx(0,c)]};
            auto [m,r]=merge_line_left(line);
            out[idx(3,c)]=m[0];out[idx(2,c)]=m[1];out[idx(1,c)]=m[2];out[idx(0,c)]=m[3]; reward+=r;
        }
    } else if (action==2) {
        for (int r=0;r<4;++r) {
            std::array<int,4> line{b[idx(r,0)],b[idx(r,1)],b[idx(r,2)],b[idx(r,3)]};
            auto [m,rr]=merge_line_left(line);
            for (int cc=0;cc<4;++cc) {
                out[idx(r,cc)] = m[cc];
            }
            reward += rr;
        }
    } else {
        for (int r=0;r<4;++r) {
            std::array<int,4> line{b[idx(r,3)],b[idx(r,2)],b[idx(r,1)],b[idx(r,0)]};
            auto [m,rr]=merge_line_left(line);
            out[idx(r,3)]=m[0];out[idx(r,2)]=m[1];out[idx(r,1)]=m[2];out[idx(r,0)]=m[3]; reward+=rr;
        }
    }
    bool moved=(out!=b);
    if (!moved) reward=0;
    return {out,reward,moved};
}

std::vector<int> legal_actions(const Board& b) {
    std::vector<int> acts; acts.reserve(4);
    for (int a=0;a<4;++a) if (apply_action(b,a).moved) acts.push_back(a);
    return acts;
}
bool has_legal_move(const Board& b) { return !legal_actions(b).empty(); }

void spawn_one(Board& b, std::mt19937& rng) {
    std::vector<int> empties; empties.reserve(16);
    for (int i=0;i<16;++i) if (!b[i]) empties.push_back(i);
    if (empties.empty()) return;
    std::uniform_int_distribution<int> pd(0,(int)empties.size()-1);
    std::uniform_real_distribution<double> pr(0.0,1.0);
    b[empties[pd(rng)]] = (pr(rng)<0.9)?1:2;
}
Board new_game(std::mt19937& rng) {
    Board b{}; b.fill(0);
    spawn_one(b,rng); spawn_one(b,rng);
    return b;
}

// ---------------------------------------------------------------------------
// Symmetry permutations (8 transforms)
// ---------------------------------------------------------------------------
std::vector<Codes> make_sym_perms() {
    std::array<int,16> base{}; for (int i=0;i<16;++i) base[i]=i;
    auto rot90  = [](const std::array<int,16>& a){
        std::array<int,16> o{};
        for (int r=0;r<4;++r) for (int c=0;c<4;++c) o[idx(r,c)]=a[idx(3-c,r)];
        return o;
    };
    auto fliplr = [](const std::array<int,16>& a){
        std::array<int,16> o{};
        for (int r=0;r<4;++r) for (int c=0;c<4;++c) o[idx(r,c)]=a[idx(r,3-c)];
        return o;
    };
    std::vector<std::array<int,16>> mats;
    mats.push_back(base);
    mats.push_back(rot90(mats.back()));
    mats.push_back(rot90(mats.back()));
    mats.push_back(rot90(mats.back()));
    auto f=fliplr(base); mats.push_back(f);
    mats.push_back(rot90(mats.back()));
    mats.push_back(rot90(mats.back()));
    mats.push_back(rot90(mats.back()));

    std::vector<Codes> perms;
    std::vector<std::array<int,16>> uniq;
    for (const auto& m : mats) {
        bool dup=false;
        for (const auto& u : uniq) if (u==m){dup=true;break;}
        if (!dup) uniq.push_back(m);
    }
    for (const auto& u : uniq) { Codes p{}; for (int i=0;i<16;++i) p[i]=u[i]; perms.push_back(p); }
    return perms;
}
const std::vector<Codes> SYM_PERMS = make_sym_perms();

// ---------------------------------------------------------------------------
// NTupleValue  (sparse hash-map implementation with optional TC)
// ---------------------------------------------------------------------------
class NTupleValue {
public:
    // --------------- Matsuzaki's 8?6-tuple patterns -------------------------
    // These are the best-known patterns for 2048/2584 (Figure 5 in paper).
    // Each index refers to a cell in the flattened 4?4 board:
    //   0  1  2  3
    //   4  5  6  7
    //   8  9 10 11
    //  12 13 14 15
    static constexpr std::array<std::array<int,TUPLE_SIZE>, 8> TUPLES = {{
        {0, 1, 2, 4, 5, 6},    // 2?3 top-left block
        {1, 2, 5, 6, 9, 13},   // diagonal band
        {0, 1, 2, 3, 4, 5},    // top two rows (left half)
        {0, 1, 5, 6, 7, 10},   // L-shape variant
        {0, 1, 2, 5, 9, 10},   // L-shape
        {0, 1, 5, 9, 13, 14},  // column + edge
        {0, 1, 5, 8, 9, 13},   // Z-shape
        {0, 1, 2, 4, 6, 10},   // sparse diagonal
    }};

    NTupleValue() : weights_(8), E_(8), A_(8) {}

    // Set optimistic default value (per-weight, before any visit).
    // Call before training begins: set_default(v_init / num_tuples).
    void set_default(float v) { default_value_ = v; }
    void set_tc(bool tc)      { use_tc_ = tc; }
    float get_default() const { return default_value_; }

    // ----------- Evaluate ------------------------------------------------
    double value(const Codes& codes) const {
        double acc = 0.0;
        for (const auto& p : SYM_PERMS) {
            Codes c{}; for (int i=0;i<16;++i) c[i]=codes[p[i]];
            acc += value_single(c);
        }
        return scale_ * (acc / static_cast<double>(SYM_PERMS.size()));
    }

    // ----------- Update --------------------------------------------------
    // Returns TD error.
    double update(const Codes& codes, double target, double alpha) {
        double pred = value(codes);
        double err  = target - pred;
        double step = alpha * err / static_cast<double>(TUPLES.size());

        for (const auto& p : SYM_PERMS) {
            Codes c{}; for (int i=0;i<16;++i) c[i]=codes[p[i]];

            if (use_tc_) {
                // Temporal Coherence: adapt step per weight
                for (size_t i=0;i<TUPLES.size();++i) {
                    uint32_t k = tuple_index(c, TUPLES[i]);
                    float& e = ensure(E_[i], k, 0.0f);
                    float& a = ensure(A_[i], k, 0.0f);
                    e += static_cast<float>(err);
                    a += static_cast<float>(std::abs(err));
                    double beta = (a != 0.0f) ? static_cast<double>(std::abs(e)) / a : 1.0;
                    ensure(weights_[i], k, default_value_) +=
                        static_cast<float>(alpha * beta * err / TUPLES.size());
                }
            } else {
                for (size_t i=0;i<TUPLES.size();++i) {
                    uint32_t k = tuple_index(c, TUPLES[i]);
                    ensure(weights_[i], k, default_value_) += static_cast<float>(step);
                }
            }
        }
        return err;
    }

    // ----------- Save / Load (NTUPLE2 format) ----------------------------
    bool save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) return false;
        out.write("NTUPLE2\0", 8);
        uint32_t nt = static_cast<uint32_t>(weights_.size());
        out.write(reinterpret_cast<const char*>(&nt),       sizeof(nt));
        out.write(reinterpret_cast<const char*>(&bias_),    sizeof(double));
        out.write(reinterpret_cast<const char*>(&scale_),   sizeof(double));
        out.write(reinterpret_cast<const char*>(&default_value_), sizeof(float));
        for (const auto& w : weights_) {
            uint64_t n = static_cast<uint64_t>(w.size());
            out.write(reinterpret_cast<const char*>(&n), sizeof(uint64_t));
            for (const auto& kv : w) {
                uint32_t key = kv.first;  float val = kv.second;
                out.write(reinterpret_cast<const char*>(&key), 4);
                out.write(reinterpret_cast<const char*>(&val), 4);
            }
        }
        return true;
    }

    // ----------- Save (NTFAST1 format, speed-priority for Python loader) -
    bool save_fbin(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) return false;

        out.write("NTFAST1\0", 8);
        uint32_t nt = static_cast<uint32_t>(weights_.size());
        out.write(reinterpret_cast<const char*>(&nt), sizeof(nt));
        out.write(reinterpret_cast<const char*>(&bias_), sizeof(double));
        out.write(reinterpret_cast<const char*>(&scale_), sizeof(double));
        out.write(reinterpret_cast<const char*>(&default_value_), sizeof(float));

        for (const auto& w : weights_) {
            uint64_t n = static_cast<uint64_t>(w.size());
            out.write(reinterpret_cast<const char*>(&n), sizeof(uint64_t));
            if (n == 0) continue;

            std::vector<std::pair<uint32_t, float>> kvs;
            kvs.reserve(static_cast<size_t>(n));
            for (const auto& kv : w) kvs.push_back(kv);
            std::sort(kvs.begin(), kvs.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            for (const auto& kv : kvs) {
                uint32_t key = kv.first;
                out.write(reinterpret_cast<const char*>(&key), sizeof(uint32_t));
            }
            for (const auto& kv : kvs) {
                float val = kv.second;
                out.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
        return true;
    }

    bool load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        char magic[8]; in.read(magic,8);
        if (!in) return false;
        std::string mg(magic,8);

        bool v2    = (mg == std::string("NTUPLE2\0",8));
        bool v1    = (mg == std::string("NTUPLE1\0",8));
        bool vfast = (mg == std::string("NTFAST1\0",8));
        if (!v1 && !v2 && !vfast) return false;

        uint32_t nt=0;
        in.read(reinterpret_cast<char*>(&nt),sizeof(uint32_t));
        if (!in || nt != weights_.size()) return false;

        in.read(reinterpret_cast<char*>(&bias_),  sizeof(double));
        in.read(reinterpret_cast<char*>(&scale_), sizeof(double));
        if (!in) return false;

        if (v2 || vfast) {
            in.read(reinterpret_cast<char*>(&default_value_), sizeof(float));
            if (!in) return false;
        } else {
            default_value_ = 0.0f;
        }

        std::vector<std::unordered_map<uint32_t,float>> loaded(nt);
        for (size_t i=0;i<nt;++i) {
            uint64_t n=0; in.read(reinterpret_cast<char*>(&n),sizeof(uint64_t));
            if (!in) return false;
            // Tuple index uses 6 cells * 5 bits => at most 2^(30) distinct keys.
            if (n > (1ULL << 30)) return false;
            loaded[i].reserve(static_cast<size_t>(n));

            if (vfast) {
                std::vector<uint32_t> keys(static_cast<size_t>(n));
                std::vector<float> vals(static_cast<size_t>(n));
                if (n > 0) {
                    in.read(reinterpret_cast<char*>(keys.data()), static_cast<std::streamsize>(n * sizeof(uint32_t)));
                    in.read(reinterpret_cast<char*>(vals.data()), static_cast<std::streamsize>(n * sizeof(float)));
                    if (!in) return false;
                }
                for (size_t j=0; j<static_cast<size_t>(n); ++j) loaded[i][keys[j]] = vals[j];
            } else {
                for (uint64_t j=0;j<n;++j) {
                    uint32_t key; float val;
                    in.read(reinterpret_cast<char*>(&key),4);
                    in.read(reinterpret_cast<char*>(&val),4);
                    if (!in) return false;
                    loaded[i][key]=val;
                }
            }
        }
        weights_ = std::move(loaded);
        return true;
    }

private:
    std::vector<std::unordered_map<uint32_t,float>> weights_;
    std::vector<std::unordered_map<uint32_t,float>> E_; // TC coherence numerator
    std::vector<std::unordered_map<uint32_t,float>> A_; // TC coherence denominator
    double bias_          = 0.0;
    double scale_         = 1.0;
    float  default_value_ = 0.0f; // optimistic init: unvisited weights return this
    bool   use_tc_        = false;

    static uint32_t tuple_index(const Codes& c, const std::array<int,TUPLE_SIZE>& t) {
        uint32_t idx=0;
        for (int p : t) { int code=c[p]; if (code>31) code=31; idx=(idx<<5)|static_cast<uint32_t>(code); }
        return idx;
    }
    double value_single(const Codes& c) const {
        double v=bias_;
        for (size_t i=0;i<TUPLES.size();++i) {
            uint32_t k=tuple_index(c,TUPLES[i]);
            auto it=weights_[i].find(k);
            v += (it!=weights_[i].end()) ? it->second : default_value_;
        }
        return v;
    }
    static float& ensure(std::unordered_map<uint32_t,float>& m, uint32_t k, float def) {
        auto it=m.find(k);
        if (it==m.end()) { m[k]=def; return m[k]; }
        return it->second;
    }
};

// ---------------------------------------------------------------------------
// Action selection  (pure TD: score = r + V(after), no heuristic bonuses)
// ---------------------------------------------------------------------------
int best_afterstate_action(const Board& board, const NTupleValue& vf,
                           MoveResult* out_after=nullptr, Codes* out_codes=nullptr) {
    auto legal = legal_actions(board);
    if (legal.empty()) return 0;

    int best_a=legal[0]; double best_s=-1e30;
    MoveResult best_mr{}; Codes best_c{};

    for (int a : legal) {
        auto mr = apply_action(board,a);
        if (!mr.moved) continue;
        auto codes = board_to_codes(mr.board);
        // Pure TD score: immediate reward + afterstate value (γ=1)
        double score = static_cast<double>(mr.reward) + vf.value(codes);
        if (score > best_s) { best_s=score; best_a=a; best_mr=mr; best_c=codes; }
    }
    if (out_after) *out_after=best_mr;
    if (out_codes) *out_codes=best_c;
    return best_a;
}

// ---------------------------------------------------------------------------
// MCTS helper (used during training when --train-use-mcts is set)
// ---------------------------------------------------------------------------
double rollout_value(const Board& board, const NTupleValue& vf, std::mt19937& rng, int depth) {
    Board cur=board; double total=0.0;
    for (int d=0;d<depth;++d) {
        MoveResult after; Codes ac;
        best_afterstate_action(cur,vf,&after,&ac);
        if (!after.moved) break;
        total += static_cast<double>(after.reward) + vf.value(ac);
        cur=after.board; spawn_one(cur,rng);
    }
    return total;
}

int mcts_afterstate_action(const Board& board, const NTupleValue& vf,
                           const Args& args, uint32_t seed,
                           MoveResult* out_after=nullptr, Codes* out_codes=nullptr) {
    auto legal=legal_actions(board);
    if (legal.empty()) return 0;
    if (legal.size()==1) {
        MoveResult a; Codes c;
        best_afterstate_action(board,vf,&a,&c);
        if (out_after) *out_after = a;
        if (out_codes) *out_codes = c;
        return legal[0];
    }

    std::vector<double> pri(legal.size(),0.0);
    std::vector<MoveResult> afters(legal.size());
    std::vector<Codes> after_codes(legal.size());
    std::vector<int> rewards(legal.size(),0);

    for (size_t i=0;i<legal.size();++i) {
        auto mr=apply_action(board,legal[i]);
        afters[i]=mr; rewards[i]=mr.reward;
        after_codes[i]=board_to_codes(mr.board);
        pri[i] = !mr.moved ? -1e30 :
                 static_cast<double>(mr.reward) + vf.value(after_codes[i]);
    }
    double mp=*std::max_element(pri.begin(),pri.end()), se=0.0;
    for (double& x:pri){x=std::exp(x-mp);se+=x;}
    if (se>0) for (double& x:pri) x/=se;

    std::vector<double> n(legal.size(),0.0), w(legal.size(),0.0);
    std::mt19937 rng(seed);

    for (int s=0;s<std::max(1,args.mcts_sims);++s) {
        double tn=std::accumulate(n.begin(),n.end(),0.0)+1.0;
        size_t bi=0; double bv=-1e30;
        for (size_t i=0;i<legal.size();++i) {
            double q=(n[i]>0.0)?(w[i]/n[i]):0.0;
            double u=args.mcts_c_puct*pri[i]*std::sqrt(tn)/(1.0+n[i]);
            if (q+u>bv){bv=q+u;bi=i;}
        }
        Board st=afters[bi].board; spawn_one(st,rng);
        double v=static_cast<double>(rewards[bi])+rollout_value(st,vf,rng,args.mcts_rollout_depth);
        n[bi]+=1.0; w[bi]+=v;
    }
    size_t pick=static_cast<size_t>(std::distance(n.begin(),std::max_element(n.begin(),n.end())));
    if (out_after) *out_after=afters[pick];
    if (out_codes) *out_codes=after_codes[pick];
    return legal[pick];
}

int select_action(const Board& board, const NTupleValue& vf, const Args& args,
                  uint32_t seed, MoveResult* out_after, Codes* out_codes) {
    int empties=0; for (int x:board) empties+=(x==0);
    if (args.train_use_mcts && empties<=args.mcts_trigger_empty_le)
        return mcts_afterstate_action(board,vf,args,seed,out_after,out_codes);
    return best_afterstate_action(board,vf,out_after,out_codes);
}

// ---------------------------------------------------------------------------
// Checkpoint paths & conversion helpers
// ---------------------------------------------------------------------------
std::string latest_bin_path(const Args& a) {
    return (fs::path(a.checkpoint_dir)/"n_tuple_weights_latest.bin").string();
}
std::string latest_fbin_path(const Args& a) {
    return (fs::path(a.checkpoint_dir)/"n_tuple_weights_latest.fbin").string();
}
bool try_convert_npz_to_bin(const std::string& npz, const std::string& bin) {
    for (const char* py : {"python3","python"}) {
        std::string cmd = std::string(py)+" ./convert_ntuple_npz_to_bin.py --input \""+npz+"\" --output \""+bin+"\"";
        if (std::system(cmd.c_str())==0) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------
double mean_last(const std::vector<int>& v, int k) {
    if (v.empty()) return 0.0;
    int n=std::min<int>(k,(int)v.size());
    double s=0.0;
    for (int i=(int)v.size()-n;i<(int)v.size();++i) s+=v[i];
    return s/n;
}
double std_last(const std::vector<int>& v, int k) {
    if (v.empty()) return 0.0;
    int n=std::min<int>(k,(int)v.size());
    double m=mean_last(v,n), s=0.0;
    for (int i=(int)v.size()-n;i<(int)v.size();++i){double d=v[i]-m;s+=d*d;}
    return std::sqrt(s/n);
}

// ---------------------------------------------------------------------------
// Dataset writer
// ---------------------------------------------------------------------------
struct TransitionRecord {
    Codes state_codes{};
    uint8_t action = 0;
    int32_t reward = 0;
    Codes next_state_codes{};
    uint8_t done = 0;
};

struct EpisodeRecord {
    int episode = 0;
    int stage = 0;
    int score = 0;
    int max_tile = 0;
    std::vector<TransitionRecord> transitions;
};

int max_tile_value(const Board& b) {
    int m = 0;
    for (int x : b) m = std::max(m, x);
    return m;
}

class DatasetWriter {
public:
    explicit DatasetWriter(const Args& args) : args_(args) {
        fs::create_directories(args_.dataset_dir);
        fs::create_directories(fs::path(args_.dataset_dir) / "trajectories");
        meta_.open(fs::path(args_.dataset_dir) / "metadata.csv", std::ios::out);
        if (!meta_) {
            std::cerr << "Error: cannot open dataset metadata for writing.\n";
            std::exit(1);
        }
        meta_ << "trajectory_id,accepted,episode,stage,score,score_bin,steps,max_tile,path\n";

        std::ofstream schema(fs::path(args_.dataset_dir) / "schema.txt", std::ios::out);
        schema
            << "2584 offline RL trajectory dataset\n"
            << "Each .bin trajectory file layout is little-endian:\n"
            << "  char[8] magic = F2584T1\\0\n"
            << "  uint32 episode, uint32 stage, int32 score, uint32 steps, int32 max_tile\n"
            << "  repeated steps times:\n"
            << "    uint8 state_codes[16], uint8 action, int32 reward,\n"
            << "    uint8 next_state_codes[16], uint8 done\n"
            << "Actions: 0=up, 1=down, 2=left, 3=right.\n"
            << "score_bin = floor(score / score_bin_width).\n";
    }

    bool accept(const EpisodeRecord& ep) {
        if (args_.collect_every <= 0) return false;
        if (ep.episode % args_.collect_every != 0) return false;
        int bin = score_bin(ep.score);
        auto key = std::make_pair(ep.stage, bin);
        int& count = counts_[key];
        if (args_.max_per_stage_bin > 0 && count >= args_.max_per_stage_bin) return false;
        ++count;
        return true;
    }

    void write(const EpisodeRecord& ep, bool accepted) {
        int bin = score_bin(ep.score);
        std::string rel_path;
        int id = next_id_++;

        if (accepted) {
            char name[128];
            std::snprintf(name, sizeof(name), "traj_%07d_ep%07d_score%06d.bin",
                          id, ep.episode, std::max(0, ep.score));
            fs::path path = fs::path(args_.dataset_dir) / "trajectories" / name;
            rel_path = (fs::path("trajectories") / name).generic_string();
            write_binary(path, ep);
        }

        if (accepted || args_.save_rejected_meta) {
            meta_ << id << ','
                  << (accepted ? 1 : 0) << ','
                  << ep.episode << ','
                  << ep.stage << ','
                  << ep.score << ','
                  << bin << ','
                  << ep.transitions.size() << ','
                  << ep.max_tile << ','
                  << rel_path << '\n';
        }
    }

private:
    Args args_;
    std::ofstream meta_;
    int next_id_ = 0;
    std::map<std::pair<int,int>, int> counts_;

    int score_bin(int score) const {
        int width = std::max(1, args_.score_bin_width);
        return std::max(0, score / width);
    }

    static void write_codes(std::ofstream& out, const Codes& c) {
        for (int x : c) {
            uint8_t v = static_cast<uint8_t>(std::clamp(x, 0, 31));
            out.write(reinterpret_cast<const char*>(&v), sizeof(uint8_t));
        }
    }

    static void write_binary(const fs::path& path, const EpisodeRecord& ep) {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            std::cerr << "Error: cannot write trajectory: " << path << "\n";
            std::exit(1);
        }

        out.write("F2584T1\0", 8);
        uint32_t episode = static_cast<uint32_t>(ep.episode);
        uint32_t stage = static_cast<uint32_t>(ep.stage);
        int32_t score = static_cast<int32_t>(ep.score);
        uint32_t steps = static_cast<uint32_t>(ep.transitions.size());
        int32_t max_tile = static_cast<int32_t>(ep.max_tile);
        out.write(reinterpret_cast<const char*>(&episode), sizeof(episode));
        out.write(reinterpret_cast<const char*>(&stage), sizeof(stage));
        out.write(reinterpret_cast<const char*>(&score), sizeof(score));
        out.write(reinterpret_cast<const char*>(&steps), sizeof(steps));
        out.write(reinterpret_cast<const char*>(&max_tile), sizeof(max_tile));

        for (const auto& tr : ep.transitions) {
            write_codes(out, tr.state_codes);
            out.write(reinterpret_cast<const char*>(&tr.action), sizeof(uint8_t));
            out.write(reinterpret_cast<const char*>(&tr.reward), sizeof(int32_t));
            write_codes(out, tr.next_state_codes);
            out.write(reinterpret_cast<const char*>(&tr.done), sizeof(uint8_t));
        }
    }
};

// ---------------------------------------------------------------------------
// Alpha schedule:  0..50% ??alpha,  50..75% ??alpha_mid,  75..100% ??alpha_final
// ---------------------------------------------------------------------------
double compute_alpha(int ep, int total, const Args& a) {
    double frac = static_cast<double>(ep) / static_cast<double>(total);
    if (frac >= 0.75) return a.alpha_final;
    if (frac >= 0.50) return a.alpha_mid;
    return a.alpha;
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
Args parse_args(int argc, char** argv) {
    Args a;
    for (int i=1;i<argc;++i) {
        std::string k=argv[i];
        auto need=[&](const std::string& key){
            if (i+1>=argc){std::cerr<<"Missing value for "<<key<<"\n";std::exit(2);}
            return std::string(argv[++i]);
        };
        if      (k=="--episodes")              a.episodes              = std::stoi(need(k));
        else if (k=="--alpha")                 a.alpha                 = std::stod(need(k));
        else if (k=="--alpha-mid")             a.alpha_mid             = std::stod(need(k));
        else if (k=="--alpha-final")           a.alpha_final           = std::stod(need(k));
        else if (k=="--v-init")                a.v_init                = std::stod(need(k));
        else if (k=="--use-tc")                a.use_tc                = true;
        else if (k=="--save-path")             a.save_path             = need(k);
        else if (k=="--save-every")            a.save_every            = std::stoi(need(k));
        else if (k=="--checkpoint-dir")        a.checkpoint_dir        = need(k);
        else if (k=="--log-every")             a.log_every             = std::stoi(need(k));
        else if (k=="--seed")                  a.seed                  = std::stoi(need(k));
        else if (k=="--continue")              a.resume                = true;
        else if (k=="--train-use-mcts")        a.train_use_mcts        = true;
        else if (k=="--no-train-mcts")         a.train_use_mcts        = false;
        else if (k=="--mcts-sims")             a.mcts_sims             = std::stoi(need(k));
        else if (k=="--mcts-c-puct")           a.mcts_c_puct           = std::stod(need(k));
        else if (k=="--mcts-rollout-depth")    a.mcts_rollout_depth    = std::stoi(need(k));
        else if (k=="--mcts-trigger-empty-le") a.mcts_trigger_empty_le = std::stoi(need(k));
        else if (k=="--dataset-dir")           a.dataset_dir           = need(k);
        else if (k=="--collect-every")         a.collect_every         = std::stoi(need(k));
        else if (k=="--stage-episodes")        a.stage_episodes        = std::stoi(need(k));
        else if (k=="--score-bin-width")       a.score_bin_width       = std::stoi(need(k));
        else if (k=="--max-per-stage-bin")     a.max_per_stage_bin     = std::stoi(need(k));
        else if (k=="--save-rejected-meta")    a.save_rejected_meta    = true;
        else { std::cerr<<"Unknown argument: "<<k<<"\n"; std::exit(2); }
    }
    return a;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    NTupleValue vf;
    vf.set_tc(args.use_tc);

    std::cout << "Train config: episodes=" << args.episodes
              << " alpha=" << args.alpha
              << " alpha_mid=" << args.alpha_mid
              << " alpha_final=" << args.alpha_final
              << " use_tc=" << (args.use_tc ? 1 : 0)
              << " train_use_mcts=" << (args.train_use_mcts ? 1 : 0)
              << " mcts_sims=" << args.mcts_sims
              << " mcts_c_puct=" << args.mcts_c_puct
              << " mcts_rollout_depth=" << args.mcts_rollout_depth
              << " mcts_trigger_empty_le=" << args.mcts_trigger_empty_le
              << " dataset_dir=" << args.dataset_dir
              << " collect_every=" << args.collect_every
              << " stage_episodes=" << args.stage_episodes
              << " score_bin_width=" << args.score_bin_width
              << " max_per_stage_bin=" << args.max_per_stage_bin
              << "\n";

    if (args.save_every>0) fs::create_directories(args.checkpoint_dir);

    // ---------- Resume from checkpoint ------------------------------------
    if (args.resume) {
        std::string latest_fast = latest_fbin_path(args);
        std::string latest = latest_bin_path(args);
        bool loaded = vf.load(latest_fast) || vf.load(latest) || vf.load(args.save_path);
        if (!loaded) {
            std::string npz = (fs::path(args.checkpoint_dir)/"n_tuple_weights_latest.npz").string();
            if (fs::exists(npz) && try_convert_npz_to_bin(npz, latest))
                loaded = vf.load(latest);
        }
        if (!loaded) {
            std::cerr << "Error: No compatible checkpoint found for resuming.\n";
            return 1;
        }
        std::cout << "Resumed. default_value=" << vf.get_default() << "\n";
    }

    // ---------- Optimistic initialisation (only on fresh start) -----------
    if (!args.resume && args.v_init > 0.0) {
        float per_weight = static_cast<float>(args.v_init / NTupleValue::TUPLES.size());
        vf.set_default(per_weight);
        std::cout << "Optimistic init: v_init=" << args.v_init
                  << "  per-weight default=" << per_weight << "\n";
    }

    // ---------- Training loop --------------------------------------------
    std::vector<int> episode_scores;
    episode_scores.reserve(args.episodes);
    ProgressBar bar(args.episodes);
    DatasetWriter dataset(args);

    for (int ep=1;ep<=args.episodes;++ep) {
        std::mt19937 ep_rng(static_cast<uint32_t>(args.seed + ep));
        Board board = new_game(ep_rng);

        int ep_score=0, step_count=0;
        double cur_alpha = compute_alpha(ep, args.episodes, args);
        EpisodeRecord record;
        record.episode = ep;
        record.stage = args.stage_episodes > 0 ? (ep - 1) / args.stage_episodes : 0;

        while (true) {
            auto legal = legal_actions(board);
            if (legal.empty()) break;

            // Select action ??get afterstate
            Codes state_codes = board_to_codes(board);
            MoveResult after; Codes after_codes;
            uint32_t step_seed = static_cast<uint32_t>(args.seed + ep*100000LL + step_count*9973 + 17);
            int action = select_action(board, vf, args, step_seed, &after, &after_codes);
            if (!after.moved) break;

            board = after.board;
            ep_score += after.reward;
            ++step_count;

            spawn_one(board, ep_rng);
            bool done = !has_legal_move(board);
            Codes next_state_codes = board_to_codes(board);

            TransitionRecord tr;
            tr.state_codes = state_codes;
            tr.action = static_cast<uint8_t>(action);
            tr.reward = static_cast<int32_t>(after.reward);
            tr.next_state_codes = next_state_codes;
            tr.done = static_cast<uint8_t>(done ? 1 : 0);
            record.transitions.push_back(tr);

            // ---- TD target: r_{t+1} + V(s'_{t+1})  (γ = 1) ---------------
            double target = 0.0;
            if (!done) {
                MoveResult next_after; Codes next_codes;
                uint32_t ns = static_cast<uint32_t>(args.seed + ep*100000LL + step_count*9973 + 7919);
                int next_action = select_action(board, vf, args, ns, &next_after, &next_codes);
                (void)next_action;
                if (next_after.moved)
                    target = static_cast<double>(next_after.reward) + vf.value(next_codes);
            }
            // ----------------------------------------------------------------

            vf.update(after_codes, target, cur_alpha);
        }

        episode_scores.push_back(ep_score);
        record.score = ep_score;
        record.max_tile = max_tile_value(board);
        bool accepted = dataset.accept(record);
        dataset.write(record, accepted);

        if (args.log_every>0 && ep%args.log_every==0) {
            std::cout << "\nEp " << std::setw(7) << ep
                      << " | avg(" << args.log_every << ")=" << std::setw(9) << std::fixed << std::setprecision(0) << mean_last(episode_scores, args.log_every)
                      << " | std=" << std::setw(8) << std::setprecision(0) << std_last(episode_scores, args.log_every)
                      << " | last=" << std::setw(8) << ep_score
                      << " | steps=" << std::setw(5) << step_count
                      << " | α=" << std::setprecision(5) << cur_alpha
                      << "\n";
        }
        if (args.save_every>0 && ep%args.save_every==0) {
            std::string latest_fast = latest_fbin_path(args);
            if (vf.save_fbin(latest_fast)) {
                std::cout << "Checkpoint saved: " << latest_fast << "\n";
            } else {
                std::cerr << "Warning: failed to save checkpoint: " << latest_fast << "\n";
            }
        }
        bar.update(ep);
    }

    bool ok = false;
    if (has_suffix(args.save_path, ".fbin")) ok = vf.save_fbin(args.save_path);
    else ok = vf.save(args.save_path);

    if (!ok) {
        std::cerr << "Error: failed to save final weights to: " << args.save_path << "\n";
        return 1;
    }
    std::cout << "Final weights saved to: " << args.save_path << "\n";
    return 0;
}

