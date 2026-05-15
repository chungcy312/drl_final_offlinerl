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
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static const size_t TUPLE_SIZE = 6;

struct Args {
    int episodes = 50000;
    double alpha = 0.1;
    double alpha_mid = 0.02;
    double alpha_final = 0.01;
    double v_init = 0.0;
    bool use_tc = false;
    std::string dataset_dir = "generate_2584_dataset/data";
    std::string save_path = "train_2584_offline/n_tuple_weights_offline.fbin";
    std::string checkpoint_dir = "train_2584_offline/checkpoints";
    int save_every = 1000;
    int log_every = 100;
    int seed = 0;
    bool resume = false;

    int eval_every = 500;
    int eval_episodes = 100;
    int eval_seed = 1000000;

    double sub_margin = 0.0;
    double sub_max_above = 5000.0;
    double sub_top_fraction = 1.0;
    int sub_min_count = 256;
    int sub_max_count = 5000;
};

static bool has_suffix(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

struct ProgressBar {
    int total;
    std::chrono::steady_clock::time_point start;
    explicit ProgressBar(int t) : total(t), start(std::chrono::steady_clock::now()) {}
    void update(int cur) {
        double ratio = total > 0 ? static_cast<double>(cur) / total : 1.0;
        int width = 28;
        int fill = static_cast<int>(ratio * width);
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        double eta = cur > 0 ? elapsed * (static_cast<double>(total - cur) / cur) : 0.0;
        auto fmt_time = [](double s) {
            int h = static_cast<int>(s) / 3600;
            int m = (static_cast<int>(s) % 3600) / 60;
            int ss = static_cast<int>(s) % 60;
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%d:%02d:%02d", h, m, ss);
            return std::string(buf);
        };
        std::cout << "\r[";
        for (int i = 0; i < width; ++i) std::cout << (i < fill ? '=' : ' ');
        std::cout << "] " << std::setw(3) << static_cast<int>(ratio * 100.0)
                  << "% " << cur << "/" << total
                  << " | elapsed " << fmt_time(elapsed)
                  << " | eta " << fmt_time(eta) << std::flush;
        if (cur == total) std::cout << "\n";
    }
};

class FibHelper {
public:
    FibHelper() {
        fib_values_ = {1, 2};
        while (fib_values_.back() < 6000000)
            fib_values_.push_back(fib_values_[fib_values_.size() - 1] + fib_values_[fib_values_.size() - 2]);
        value_to_code_[0] = 0;
        code_to_value_[0] = 0;
        for (size_t i = 0; i < fib_values_.size(); ++i) {
            int code = static_cast<int>(i + 1);
            value_to_code_[fib_values_[i]] = code;
            code_to_value_[code] = fib_values_[i];
        }
    }
    bool mergeable_value(int a, int b) const {
        if (a <= 0 || b <= 0) return false;
        if (a == 1 && b == 1) return true;
        auto ia = value_to_code_.find(a), ib = value_to_code_.find(b);
        if (ia == value_to_code_.end() || ib == value_to_code_.end()) return false;
        return std::abs(ia->second - ib->second) == 1;
    }
    int value_to_code(int v) const {
        auto it = value_to_code_.find(v);
        return it == value_to_code_.end() ? 0 : it->second;
    }
    int code_to_value(int code) const {
        auto it = code_to_value_.find(code);
        return it == code_to_value_.end() ? 0 : it->second;
    }
private:
    std::vector<int> fib_values_;
    std::unordered_map<int, int> value_to_code_;
    std::unordered_map<int, int> code_to_value_;
};
static const FibHelper FIB;

using Board = std::array<int, 16>;
using Codes = std::array<int, 16>;

struct MoveResult {
    Board board{};
    int reward = 0;
    bool moved = false;
};

static inline int idx(int r, int c) { return r * 4 + c; }

Codes board_to_codes(const Board& b) {
    Codes c{};
    for (int i = 0; i < 16; ++i) c[i] = FIB.value_to_code(b[i]);
    return c;
}

Board codes_to_board(const Codes& c) {
    Board b{};
    for (int i = 0; i < 16; ++i) b[i] = FIB.code_to_value(c[i]);
    return b;
}

std::pair<std::array<int, 4>, int> merge_line_left(const std::array<int, 4>& line) {
    std::vector<int> nz;
    nz.reserve(4);
    for (int x : line) if (x) nz.push_back(x);
    std::array<int, 4> out{0, 0, 0, 0};
    int top = 0, reward = 0;
    size_t i = 0;
    while (i < nz.size()) {
        if (i + 1 < nz.size() && FIB.mergeable_value(nz[i], nz[i + 1])) {
            out[top++] = nz[i] + nz[i + 1];
            reward += nz[i] + nz[i + 1];
            i += 2;
        } else {
            out[top++] = nz[i++];
        }
    }
    return {out, reward};
}

MoveResult apply_action(const Board& b, int action) {
    Board out{};
    int reward = 0;
    if (action == 0) {
        for (int c = 0; c < 4; ++c) {
            std::array<int, 4> line{b[idx(0,c)], b[idx(1,c)], b[idx(2,c)], b[idx(3,c)]};
            auto [m, r] = merge_line_left(line);
            for (int rr = 0; rr < 4; ++rr) out[idx(rr, c)] = m[rr];
            reward += r;
        }
    } else if (action == 1) {
        for (int c = 0; c < 4; ++c) {
            std::array<int, 4> line{b[idx(3,c)], b[idx(2,c)], b[idx(1,c)], b[idx(0,c)]};
            auto [m, r] = merge_line_left(line);
            out[idx(3,c)] = m[0]; out[idx(2,c)] = m[1]; out[idx(1,c)] = m[2]; out[idx(0,c)] = m[3];
            reward += r;
        }
    } else if (action == 2) {
        for (int r = 0; r < 4; ++r) {
            std::array<int, 4> line{b[idx(r,0)], b[idx(r,1)], b[idx(r,2)], b[idx(r,3)]};
            auto [m, rr] = merge_line_left(line);
            for (int c = 0; c < 4; ++c) out[idx(r, c)] = m[c];
            reward += rr;
        }
    } else {
        for (int r = 0; r < 4; ++r) {
            std::array<int, 4> line{b[idx(r,3)], b[idx(r,2)], b[idx(r,1)], b[idx(r,0)]};
            auto [m, rr] = merge_line_left(line);
            out[idx(r,3)] = m[0]; out[idx(r,2)] = m[1]; out[idx(r,1)] = m[2]; out[idx(r,0)] = m[3];
            reward += rr;
        }
    }
    bool moved = out != b;
    if (!moved) reward = 0;
    return {out, reward, moved};
}

std::vector<int> legal_actions(const Board& b) {
    std::vector<int> acts;
    acts.reserve(4);
    for (int a = 0; a < 4; ++a) if (apply_action(b, a).moved) acts.push_back(a);
    return acts;
}

bool has_legal_move(const Board& b) { return !legal_actions(b).empty(); }

void spawn_one(Board& b, std::mt19937& rng) {
    std::vector<int> empties;
    empties.reserve(16);
    for (int i = 0; i < 16; ++i) if (!b[i]) empties.push_back(i);
    if (empties.empty()) return;
    std::uniform_int_distribution<int> pd(0, static_cast<int>(empties.size()) - 1);
    std::uniform_real_distribution<double> pr(0.0, 1.0);
    b[empties[pd(rng)]] = (pr(rng) < 0.9) ? 1 : 2;
}

Board new_game(std::mt19937& rng) {
    Board b{};
    b.fill(0);
    spawn_one(b, rng);
    spawn_one(b, rng);
    return b;
}

std::vector<Codes> make_sym_perms() {
    std::array<int,16> base{};
    for (int i = 0; i < 16; ++i) base[i] = i;
    auto rot90 = [](const std::array<int,16>& a) {
        std::array<int,16> o{};
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) o[idx(r,c)] = a[idx(3-c,r)];
        return o;
    };
    auto fliplr = [](const std::array<int,16>& a) {
        std::array<int,16> o{};
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) o[idx(r,c)] = a[idx(r,3-c)];
        return o;
    };
    std::vector<std::array<int,16>> mats;
    mats.push_back(base);
    mats.push_back(rot90(mats.back()));
    mats.push_back(rot90(mats.back()));
    mats.push_back(rot90(mats.back()));
    mats.push_back(fliplr(base));
    mats.push_back(rot90(mats.back()));
    mats.push_back(rot90(mats.back()));
    mats.push_back(rot90(mats.back()));
    std::vector<std::array<int,16>> uniq;
    for (const auto& m : mats) {
        bool dup = false;
        for (const auto& u : uniq) if (u == m) { dup = true; break; }
        if (!dup) uniq.push_back(m);
    }
    std::vector<Codes> perms;
    for (const auto& u : uniq) {
        Codes p{};
        for (int i = 0; i < 16; ++i) p[i] = u[i];
        perms.push_back(p);
    }
    return perms;
}
const std::vector<Codes> SYM_PERMS = make_sym_perms();

class NTupleValue {
public:
    static constexpr std::array<std::array<int,TUPLE_SIZE>, 8> TUPLES = {{
        {0, 1, 2, 4, 5, 6},
        {1, 2, 5, 6, 9, 13},
        {0, 1, 2, 3, 4, 5},
        {0, 1, 5, 6, 7, 10},
        {0, 1, 2, 5, 9, 10},
        {0, 1, 5, 9, 13, 14},
        {0, 1, 5, 8, 9, 13},
        {0, 1, 2, 4, 6, 10},
    }};

    NTupleValue() : weights_(8), E_(8), A_(8) {}
    void set_default(float v) { default_value_ = v; }
    void set_tc(bool tc) { use_tc_ = tc; }
    float get_default() const { return default_value_; }

    double value(const Codes& codes) const {
        double acc = 0.0;
        for (const auto& p : SYM_PERMS) {
            Codes c{};
            for (int i = 0; i < 16; ++i) c[i] = codes[p[i]];
            acc += value_single(c);
        }
        return scale_ * (acc / static_cast<double>(SYM_PERMS.size()));
    }

    double update(const Codes& codes, double target, double alpha) {
        double pred = value(codes);
        double err = target - pred;
        double step = alpha * err / static_cast<double>(TUPLES.size());
        for (const auto& p : SYM_PERMS) {
            Codes c{};
            for (int i = 0; i < 16; ++i) c[i] = codes[p[i]];
            if (use_tc_) {
                for (size_t i = 0; i < TUPLES.size(); ++i) {
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
                for (size_t i = 0; i < TUPLES.size(); ++i) {
                    uint32_t k = tuple_index(c, TUPLES[i]);
                    ensure(weights_[i], k, default_value_) += static_cast<float>(step);
                }
            }
        }
        return err;
    }

    bool save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) return false;
        out.write("NTUPLE2\0", 8);
        uint32_t nt = static_cast<uint32_t>(weights_.size());
        out.write(reinterpret_cast<const char*>(&nt), sizeof(nt));
        out.write(reinterpret_cast<const char*>(&bias_), sizeof(double));
        out.write(reinterpret_cast<const char*>(&scale_), sizeof(double));
        out.write(reinterpret_cast<const char*>(&default_value_), sizeof(float));
        for (const auto& w : weights_) {
            uint64_t n = static_cast<uint64_t>(w.size());
            out.write(reinterpret_cast<const char*>(&n), sizeof(uint64_t));
            for (const auto& kv : w) {
                uint32_t key = kv.first;
                float val = kv.second;
                out.write(reinterpret_cast<const char*>(&key), sizeof(uint32_t));
                out.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
        return true;
    }

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
            std::vector<std::pair<uint32_t, float>> kvs;
            kvs.reserve(static_cast<size_t>(n));
            for (const auto& kv : w) kvs.push_back(kv);
            std::sort(kvs.begin(), kvs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& kv : kvs) out.write(reinterpret_cast<const char*>(&kv.first), sizeof(uint32_t));
            for (const auto& kv : kvs) out.write(reinterpret_cast<const char*>(&kv.second), sizeof(float));
        }
        return true;
    }

    bool load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        char magic[8];
        in.read(magic, 8);
        std::string mg(magic, 8);
        bool v2 = mg == std::string("NTUPLE2\0", 8);
        bool v1 = mg == std::string("NTUPLE1\0", 8);
        bool vfast = mg == std::string("NTFAST1\0", 8);
        if (!v1 && !v2 && !vfast) return false;
        uint32_t nt = 0;
        in.read(reinterpret_cast<char*>(&nt), sizeof(uint32_t));
        if (!in || nt != weights_.size()) return false;
        in.read(reinterpret_cast<char*>(&bias_), sizeof(double));
        in.read(reinterpret_cast<char*>(&scale_), sizeof(double));
        if (v2 || vfast) in.read(reinterpret_cast<char*>(&default_value_), sizeof(float));
        else default_value_ = 0.0f;
        if (!in) return false;
        std::vector<std::unordered_map<uint32_t, float>> loaded(nt);
        for (size_t i = 0; i < nt; ++i) {
            uint64_t n = 0;
            in.read(reinterpret_cast<char*>(&n), sizeof(uint64_t));
            if (!in || n > (1ULL << 30)) return false;
            loaded[i].reserve(static_cast<size_t>(n));
            if (vfast) {
                std::vector<uint32_t> keys(static_cast<size_t>(n));
                std::vector<float> vals(static_cast<size_t>(n));
                if (n > 0) {
                    in.read(reinterpret_cast<char*>(keys.data()), static_cast<std::streamsize>(n * sizeof(uint32_t)));
                    in.read(reinterpret_cast<char*>(vals.data()), static_cast<std::streamsize>(n * sizeof(float)));
                    if (!in) return false;
                }
                for (size_t j = 0; j < static_cast<size_t>(n); ++j) loaded[i][keys[j]] = vals[j];
            } else {
                for (uint64_t j = 0; j < n; ++j) {
                    uint32_t key = 0;
                    float val = 0.0f;
                    in.read(reinterpret_cast<char*>(&key), sizeof(uint32_t));
                    in.read(reinterpret_cast<char*>(&val), sizeof(float));
                    if (!in) return false;
                    loaded[i][key] = val;
                }
            }
        }
        weights_ = std::move(loaded);
        return true;
    }

private:
    std::vector<std::unordered_map<uint32_t, float>> weights_;
    std::vector<std::unordered_map<uint32_t, float>> E_;
    std::vector<std::unordered_map<uint32_t, float>> A_;
    double bias_ = 0.0;
    double scale_ = 1.0;
    float default_value_ = 0.0f;
    bool use_tc_ = false;

    static uint32_t tuple_index(const Codes& c, const std::array<int,TUPLE_SIZE>& t) {
        uint32_t out = 0;
        for (int p : t) {
            int code = c[p];
            if (code > 31) code = 31;
            out = (out << 5) | static_cast<uint32_t>(code);
        }
        return out;
    }
    double value_single(const Codes& c) const {
        double v = bias_;
        for (size_t i = 0; i < TUPLES.size(); ++i) {
            uint32_t k = tuple_index(c, TUPLES[i]);
            auto it = weights_[i].find(k);
            v += it != weights_[i].end() ? it->second : default_value_;
        }
        return v;
    }
    static float& ensure(std::unordered_map<uint32_t, float>& m, uint32_t k, float def) {
        auto it = m.find(k);
        if (it == m.end()) {
            auto inserted = m.emplace(k, def);
            return inserted.first->second;
        }
        return it->second;
    }
};

int best_afterstate_action(const Board& board, const NTupleValue& vf,
                           MoveResult* out_after = nullptr, Codes* out_codes = nullptr) {
    auto legal = legal_actions(board);
    if (legal.empty()) return 0;
    int best_a = legal[0];
    double best_s = -1e30;
    MoveResult best_mr{};
    Codes best_c{};
    for (int a : legal) {
        auto mr = apply_action(board, a);
        if (!mr.moved) continue;
        auto codes = board_to_codes(mr.board);
        double score = static_cast<double>(mr.reward) + vf.value(codes);
        if (score > best_s) {
            best_s = score;
            best_a = a;
            best_mr = mr;
            best_c = codes;
        }
    }
    if (out_after) *out_after = best_mr;
    if (out_codes) *out_codes = best_c;
    return best_a;
}

struct TrajectoryIndex {
    fs::path path;
    uint32_t episode = 0;
    uint32_t stage = 0;
    int32_t score = 0;
    uint32_t steps = 0;
    int32_t max_tile = 0;
};

struct Transition {
    Codes state_codes{};
    uint8_t action = 0;
    int32_t reward = 0;
    Codes next_state_codes{};
    uint8_t done = 0;
};

struct Trajectory {
    TrajectoryIndex meta;
    std::vector<Transition> transitions;
};

template <typename T>
bool read_exact(std::ifstream& in, T& v) {
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    return static_cast<bool>(in);
}

TrajectoryIndex read_header(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open trajectory: " + path.string());
    char magic[8];
    in.read(magic, 8);
    if (std::string(magic, 8) != std::string("F2584T1\0", 8))
        throw std::runtime_error("bad trajectory magic: " + path.string());
    TrajectoryIndex meta;
    meta.path = path;
    if (!read_exact(in, meta.episode) || !read_exact(in, meta.stage) ||
        !read_exact(in, meta.score) || !read_exact(in, meta.steps) ||
        !read_exact(in, meta.max_tile))
        throw std::runtime_error("short trajectory header: " + path.string());
    return meta;
}

Trajectory load_trajectory(const TrajectoryIndex& meta) {
    std::ifstream in(meta.path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open trajectory: " + meta.path.string());
    char magic[8];
    in.read(magic, 8);
    if (std::string(magic, 8) != std::string("F2584T1\0", 8))
        throw std::runtime_error("bad trajectory magic: " + meta.path.string());
    Trajectory tr;
    tr.meta = meta;
    uint32_t episode = 0, stage = 0, steps = 0;
    int32_t score = 0, max_tile = 0;
    read_exact(in, episode); read_exact(in, stage); read_exact(in, score);
    read_exact(in, steps); read_exact(in, max_tile);
    tr.transitions.resize(steps);
    for (auto& t : tr.transitions) {
        for (int i = 0; i < 16; ++i) {
            uint8_t x = 0;
            read_exact(in, x);
            t.state_codes[i] = x;
        }
        read_exact(in, t.action);
        read_exact(in, t.reward);
        for (int i = 0; i < 16; ++i) {
            uint8_t x = 0;
            read_exact(in, x);
            t.next_state_codes[i] = x;
        }
        read_exact(in, t.done);
        if (!in) throw std::runtime_error("short trajectory body: " + meta.path.string());
    }
    return tr;
}

std::vector<TrajectoryIndex> scan_dataset(const std::string& dataset_dir) {
    fs::path traj_dir = fs::path(dataset_dir) / "trajectories";
    if (!fs::exists(traj_dir)) throw std::runtime_error("missing trajectories dir: " + traj_dir.string());
    std::vector<fs::path> paths;
    for (const auto& ent : fs::directory_iterator(traj_dir)) {
        if (ent.is_regular_file() && ent.path().extension() == ".bin")
            paths.push_back(ent.path());
    }
    std::sort(paths.begin(), paths.end());

    std::vector<TrajectoryIndex> out;
    out.reserve(paths.size());
    ProgressBar bar(static_cast<int>(paths.size()));
    for (size_t i = 0; i < paths.size(); ++i) {
        out.push_back(read_header(paths[i]));
        if ((i + 1) % 256 == 0 || i + 1 == paths.size())
            bar.update(static_cast<int>(i + 1));
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        if (a.score != b.score) return a.score < b.score;
        return a.path.string() < b.path.string();
    });
    return out;
}

double compute_alpha(int ep, int total, const Args& a) {
    double frac = static_cast<double>(ep) / static_cast<double>(std::max(1, total));
    if (frac >= 0.75) return a.alpha_final;
    if (frac >= 0.50) return a.alpha_mid;
    return a.alpha;
}

double mean_last(const std::vector<double>& v, int k) {
    if (v.empty()) return 0.0;
    int n = std::min<int>(k, static_cast<int>(v.size()));
    double s = 0.0;
    for (int i = static_cast<int>(v.size()) - n; i < static_cast<int>(v.size()); ++i) s += v[i];
    return s / n;
}

double evaluate_model(const NTupleValue& vf, int episodes, int seed) {
    if (episodes <= 0) return 0.0;
    double total = 0.0;
    for (int ep = 0; ep < episodes; ++ep) {
        std::mt19937 rng(static_cast<uint32_t>(seed + ep));
        Board board = new_game(rng);
        int score = 0;
        while (true) {
            MoveResult after;
            Codes after_codes;
            best_afterstate_action(board, vf, &after, &after_codes);
            if (!after.moved) break;
            board = after.board;
            score += after.reward;
            spawn_one(board, rng);
            if (!has_legal_move(board)) break;
        }
        total += score;
    }
    return total / static_cast<double>(episodes);
}

std::vector<size_t> select_subdataset(const std::vector<TrajectoryIndex>& data,
                                      double eval_score,
                                      const Args& args) {
    std::vector<size_t> ids;
    double lower = eval_score + args.sub_margin;
    double upper = lower + args.sub_max_above;
    bool use_upper = args.sub_max_above > 0.0;

    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i].score < lower) continue;
        if (use_upper && data[i].score > upper) continue;
        ids.push_back(i);
    }

    if (static_cast<int>(ids.size()) < args.sub_min_count && use_upper) {
        ids.clear();
        for (size_t i = 0; i < data.size(); ++i)
            if (data[i].score >= lower) ids.push_back(i);
    }

    if (static_cast<int>(ids.size()) < args.sub_min_count) {
        ids.clear();
        for (size_t i = 0; i < data.size(); ++i)
            if (data[i].score >= eval_score) ids.push_back(i);
    }

    if (ids.empty()) {
        size_t take = static_cast<size_t>(std::min<int>(std::max(1, args.sub_min_count), static_cast<int>(data.size())));
        for (size_t i = data.size() - take; i < data.size(); ++i) ids.push_back(i);
    }

    std::sort(ids.begin(), ids.end(), [&](size_t a, size_t b) {
        if (data[a].score != data[b].score) return data[a].score < data[b].score;
        return data[a].path.string() < data[b].path.string();
    });

    if (args.sub_top_fraction > 0.0 && args.sub_top_fraction < 1.0 && !ids.empty()) {
        size_t keep = std::max<size_t>(1, static_cast<size_t>(std::ceil(ids.size() * args.sub_top_fraction)));
        ids.resize(keep);
    }

    if (args.sub_max_count > 0 && static_cast<int>(ids.size()) > args.sub_max_count)
        ids.resize(static_cast<size_t>(args.sub_max_count));

    return ids;
}

std::string latest_fbin_path(const Args& a) {
    return (fs::path(a.checkpoint_dir) / "n_tuple_weights_latest.fbin").string();
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const std::string& key) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << key << "\n";
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if      (k == "--episodes")          a.episodes = std::stoi(need(k));
        else if (k == "--alpha")             a.alpha = std::stod(need(k));
        else if (k == "--alpha-mid")         a.alpha_mid = std::stod(need(k));
        else if (k == "--alpha-final")       a.alpha_final = std::stod(need(k));
        else if (k == "--v-init")            a.v_init = std::stod(need(k));
        else if (k == "--use-tc")            a.use_tc = true;
        else if (k == "--dataset-dir")       a.dataset_dir = need(k);
        else if (k == "--save-path")         a.save_path = need(k);
        else if (k == "--checkpoint-dir")    a.checkpoint_dir = need(k);
        else if (k == "--save-every")        a.save_every = std::stoi(need(k));
        else if (k == "--log-every")         a.log_every = std::stoi(need(k));
        else if (k == "--seed")              a.seed = std::stoi(need(k));
        else if (k == "--continue")          a.resume = true;
        else if (k == "--eval-every")        a.eval_every = std::stoi(need(k));
        else if (k == "--eval-episodes")     a.eval_episodes = std::stoi(need(k));
        else if (k == "--eval-seed")         a.eval_seed = std::stoi(need(k));
        else if (k == "--sub-margin")        a.sub_margin = std::stod(need(k));
        else if (k == "--sub-max-above")     a.sub_max_above = std::stod(need(k));
        else if (k == "--sub-top-fraction")  a.sub_top_fraction = std::stod(need(k));
        else if (k == "--sub-min-count")     a.sub_min_count = std::stoi(need(k));
        else if (k == "--sub-max-count")     a.sub_max_count = std::stoi(need(k));
        else {
            std::cerr << "Unknown argument: " << k << "\n";
            std::exit(2);
        }
    }
    return a;
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        fs::create_directories(args.checkpoint_dir);
        fs::create_directories(fs::path(args.save_path).parent_path());

        std::cout << "Scanning dataset: " << args.dataset_dir << "\n";
        auto dataset = scan_dataset(args.dataset_dir);
        if (dataset.empty()) {
            std::cerr << "No trajectory .bin files found.\n";
            return 1;
        }
        std::cout << "Loaded trajectory index: " << dataset.size()
                  << " trajectories | score range "
                  << dataset.front().score << ".." << dataset.back().score << "\n";

        NTupleValue vf;
        vf.set_tc(args.use_tc);
        if (args.resume) {
            bool loaded = vf.load(latest_fbin_path(args)) || vf.load(args.save_path);
            if (!loaded) {
                std::cerr << "Error: no compatible checkpoint found for --continue.\n";
                return 1;
            }
            std::cout << "Resumed. default_value=" << vf.get_default() << "\n";
        } else if (args.v_init > 0.0) {
            float per_weight = static_cast<float>(args.v_init / NTupleValue::TUPLES.size());
            vf.set_default(per_weight);
            std::cout << "Optimistic init: v_init=" << args.v_init
                      << " per-weight default=" << per_weight << "\n";
        }

        std::cout << "Offline config: episodes=" << args.episodes
                  << " alpha=" << args.alpha
                  << " alpha_mid=" << args.alpha_mid
                  << " alpha_final=" << args.alpha_final
                  << " eval_every=" << args.eval_every
                  << " eval_episodes=" << args.eval_episodes
                  << " sub_margin=" << args.sub_margin
                  << " sub_max_above=" << args.sub_max_above
                  << " sub_top_fraction=" << args.sub_top_fraction
                  << " sub_min_count=" << args.sub_min_count
                  << " sub_max_count=" << args.sub_max_count
                  << "\n";

        double eval_score = evaluate_model(vf, args.eval_episodes, args.eval_seed);
        auto pool = select_subdataset(dataset, eval_score, args);
        std::cout << "Initial eval_mean=" << std::fixed << std::setprecision(1) << eval_score
                  << " | subdataset=" << pool.size()
                  << " | score range " << dataset[pool.front()].score
                  << ".." << dataset[pool.back()].score << "\n";

        std::mt19937 rng(static_cast<uint32_t>(args.seed));
        std::vector<double> train_scores;
        train_scores.reserve(args.episodes);
        ProgressBar bar(args.episodes);

        for (int ep = 1; ep <= args.episodes; ++ep) {
            if (args.eval_every > 0 && ep > 1 && (ep - 1) % args.eval_every == 0) {
                eval_score = evaluate_model(vf, args.eval_episodes, args.eval_seed + ep * 17);
                pool = select_subdataset(dataset, eval_score, args);
                std::cout << "\nEval at offline_ep " << (ep - 1)
                          << " | eval_mean=" << std::fixed << std::setprecision(1) << eval_score
                          << " | subdataset=" << pool.size()
                          << " | score range " << dataset[pool.front()].score
                          << ".." << dataset[pool.back()].score << "\n";
            }

            std::uniform_int_distribution<size_t> pick(0, pool.size() - 1);
            const auto& meta = dataset[pool[pick(rng)]];
            Trajectory tr = load_trajectory(meta);
            double cur_alpha = compute_alpha(ep, args.episodes, args);
            double abs_err_sum = 0.0;

            for (const auto& step : tr.transitions) {
                Board state = codes_to_board(step.state_codes);
                MoveResult after = apply_action(state, static_cast<int>(step.action));
                if (!after.moved) continue;
                Codes after_codes = board_to_codes(after.board);

                double target = static_cast<double>(step.reward);
                if (!step.done) {
                    Board next_board = codes_to_board(step.next_state_codes);
                    MoveResult next_after;
                    Codes next_after_codes;
                    best_afterstate_action(next_board, vf, &next_after, &next_after_codes);
                    if (next_after.moved)
                        target += vf.value(next_after_codes);
                }
                double err = vf.update(after_codes, target, cur_alpha);
                abs_err_sum += std::abs(err);
            }

            train_scores.push_back(static_cast<double>(tr.meta.score));
            if (args.log_every > 0 && ep % args.log_every == 0) {
                double mean_score = mean_last(train_scores, args.log_every);
                double mean_abs_err = tr.transitions.empty() ? 0.0 : abs_err_sum / static_cast<double>(tr.transitions.size());
                std::cout << "\nOffline ep " << std::setw(7) << ep
                          << " | sampled_avg_score=" << std::setw(9) << std::fixed << std::setprecision(0) << mean_score
                          << " | last_score=" << std::setw(8) << tr.meta.score
                          << " | steps=" << std::setw(5) << tr.transitions.size()
                          << " | mean_abs_td=" << std::setw(10) << std::setprecision(1) << mean_abs_err
                          << " | alpha=" << std::setprecision(5) << cur_alpha
                          << "\n";
            }

            if (args.save_every > 0 && ep % args.save_every == 0) {
                if (vf.save_fbin(latest_fbin_path(args)))
                    std::cout << "Checkpoint saved: " << latest_fbin_path(args) << "\n";
                else
                    std::cerr << "Warning: failed to save checkpoint.\n";
            }
            bar.update(ep);
        }

        bool ok = has_suffix(args.save_path, ".fbin") ? vf.save_fbin(args.save_path) : vf.save(args.save_path);
        if (!ok) {
            std::cerr << "Error: failed to save final weights to " << args.save_path << "\n";
            return 1;
        }
        double final_eval = evaluate_model(vf, args.eval_episodes, args.eval_seed + args.episodes * 31);
        std::cout << "Final eval_mean=" << std::fixed << std::setprecision(1) << final_eval << "\n";
        std::cout << "Final weights saved to: " << args.save_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
