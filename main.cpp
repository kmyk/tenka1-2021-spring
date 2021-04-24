#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>
#define REP(i, n) for (int i = 0; (i) < (int)(n); ++ (i))
#define REP3(i, m, n) for (int i = (m); (i) < (int)(n); ++ (i))
#define REP_R(i, n) for (int i = (int)(n) - 1; (i) >= 0; -- (i))
#define REP3R(i, m, n) for (int i = (int)(n) - 1; (i) >= (int)(m); -- (i))
#define ALL(x) std::begin(x), std::end(x)
using namespace std;

constexpr int MAX_LEN_TASK = 10;
constexpr int NUM_AGENT = 5;
constexpr int AREA_SIZE = 30;

struct MasterData {
    int game_period;
    int max_len_task;
    int num_agent;
    array<pair<int, int>, 26> checkpoints;
    int area_size;
};

struct AgentMove {
    double x;
    double y;
    int t;
};

struct Agent {
    vector<AgentMove> move;
    string history;
};

struct Task {
    string s;
    int t;
    int weight;
    int count;
    int total;
};

struct Game {
    int now;
    array<Agent, NUM_AGENT> agent;
    vector<Task> task;
    int next_task;
};

struct Move {
    int now;
    vector<AgentMove> move;
};

MasterData call_master_data() {
    cout << "master_data" << endl;
    MasterData res;
    cin >> res.game_period >> res.max_len_task >> res.num_agent;
    for (auto& c : res.checkpoints) {
        cin >> c.first >> c.second;
    }
    cin >> res.area_size;
    assert (res.max_len_task == MAX_LEN_TASK);
    assert (res.num_agent == NUM_AGENT);
    assert (res.area_size == AREA_SIZE);
    return res;
}

Game call_game() {
    cout << "game" << endl;
    Game res;
    int num_agent, num_task;
    cin >> res.now >> num_agent >> num_task;
    for (auto& a : res.agent) {
        int num_move;
        cin >> num_move;
        a.move.resize(num_move);
        for (auto& m : a.move) {
            cin >> m.x >> m.y >> m.t;
        }
        string h;
        cin >> h;
        a.history = h.substr(1);
    }
    res.task.resize(num_task);
    for (auto& t : res.task) {
        cin >> t.s >> t.t >> t.weight >> t.count >> t.total;
    }
    cin >> res.next_task;
    return res;
}

Move read_move() {
    Move res;
    int num_move;
    cin >> res.now >> num_move;
    res.move.resize(num_move);
    for (auto& m : res.move) {
        cin >> m.x >> m.y >> m.t;
    }
    return res;
}

Move call_move(int index, int x, int y) {
    assert (1 <= index and index <= NUM_AGENT);
    assert (0 <= x and x <= AREA_SIZE);
    assert (0 <= y and y <= AREA_SIZE);
    cout << "move " << index << "-" << x << "-" << y << endl;
    return read_move();
}

Move call_move_next(int index, int x, int y) {
    assert (1 <= index and index <= 5);
    assert (0 <= x and x <= AREA_SIZE);
    assert (0 <= y and y <= AREA_SIZE);
    cout << "move_next " << index << "-" << x << "-" << y << endl;
    return read_move();
}

constexpr int GAME_INFO_SLEEP_TIME = 5000;

struct Bot {
    mt19937 gen;
    MasterData master_data;
    deque<Game> game_info;
    chrono::system_clock::time_point start_time;
    int start_game_time_ms;
    int next_call_game_info_time_ms;
    array<int, NUM_AGENT> agent_move_finish_ms;
    array<queue<pair<int, int>>, NUM_AGENT> agent_move_point_queue;
    array<deque<char>, NUM_AGENT> agent_move_history;

    Bot(mt19937& gen_)
            : gen(gen_) {
        master_data = call_master_data();
        game_info.push_back(call_game());
        start_game_time_ms = game_info.back().now;
        cerr << "Start:" << start_game_time_ms << endl;
        start_time = chrono::system_clock::now();
        next_call_game_info_time_ms = get_now_game_time_ms() + GAME_INFO_SLEEP_TIME;
        fill(ALL(agent_move_finish_ms), 0);
        REP (i, NUM_AGENT) {
            agent_move_history[i].insert(agent_move_history[i].end(), ALL(game_info.back().agent[i].history));
            set_move_point(i);
        }
    }

    const int calculate_required_time_for_move(pair<int, int> src, pair<int, int> dst) {
        // 連続して移動するときの誤差は無視する
        int dx = dst.first - src.first;
        int dy = dst.second - src.second;
        return ceil(100 * sqrt(dx * dx + dy * dy));
    }

    const int calculate_required_time_between_checkpoints(char src, char dst) {
        return calculate_required_time_for_move(get_checkpoint(src), get_checkpoint(dst));
    }

    const int calculate_required_time_for_task(int index, const Task& task) {
        assert (not agent_move_history[index].empty());
        auto current_point = master_data.checkpoints[agent_move_history[index].back() - 'A'];
        int required_time = 0;
        for (char c : task.s) {
            auto next_point = get_checkpoint(c);
            required_time += calculate_required_time_for_move(current_point, next_point);
            current_point = next_point;
        }
        return required_time;
    }

    const double predict_score(int task_index) {
        // 自分の達成による過去の得点の減少分は無視する
        const Task& task = game_info.back().task[task_index];
        int remaining_time = master_data.game_period - game_info.back().now;
        int previous_total = task_index < game_info.front().task.size() ? game_info.front().task[task_index].total : 0;
        int previous_time = task_index < game_info.front().task.size() ? game_info.front().task[task_index].t : task.t;
        double predicted_speed = (double)(task.total - previous_total) / (game_info.back().now - previous_time + 1) * 0.7;
        double predicted_total = task.total + predicted_speed * remaining_time + 1;
        return (double)task.weight / predicted_total;
    }

    const double calculate_current_score(const Task& task) {
        if (task.total == 0) return INFINITY;
        return (double)task.weight / task.total;
    }

    pair<int, int> get_checkpoint(char c) {
        return master_data.checkpoints[c - 'A'];
    }

    // 移動予定を設定
    void set_move_point(int index) {
        assert (not agent_move_history[index].empty());

        // ビームサーチ
        auto compare_state = [](tuple<string, int, double> a, tuple<string, int, double> b) {
            return get<2>(a) / (get<1>(a) + 100) > get<2>(b) / (get<1>(b) + 100);
        };
        vector<tuple<string, int, double> > cur;
        cur.emplace_back(string(ALL(agent_move_history[index])), 0, 0);
        tuple<string, int, double> best_state = cur.back();

        REP (iteration, 15) {
            vector<tuple<string, int, double> > prv;
            prv.swap(cur);
            for (auto&& state : prv) {
                auto&& [s, time, score] = state;
                if (s.length() >= MAX_LEN_TASK * 10) {
                    if (compare_state(state, best_state)) {
                        best_state = state;
                    }
                    continue;
                }
                REP (task_index, game_info.back().task.size()) {
                    const Task& task = game_info.back().task[task_index];
                    if (task.s == "K") continue;
                    REP (m, task.s.size()) {
                        if (s.size() >= m and s.compare((int)s.size() - m, m, task.s, 0, m) == 0) {
                            int time_delta = 0;
                            char b = s.back();
                            for (char c : task.s.substr(m)) {
                                time_delta += calculate_required_time_between_checkpoints(b, c);
                                b = c;
                            }
                            double score_delta = predict_score(task_index);
                            cur.emplace_back(s + task.s.substr(m), time + time_delta, score + score_delta);
                        }
                    }
                }
            }

            constexpr int WIDTH = 100;
            sort(ALL(cur), compare_state);
            if (cur.size() >= WIDTH) {
                cur.resize(WIDTH);
            }
        }
        for (auto&& state : cur) {
            if (compare_state(state, best_state)) {
                best_state = state;
            }
        }

        string s = get<0>(best_state).substr(agent_move_history[index].size());
        if (s.size() >= MAX_LEN_TASK * 3) {
            s.resize(MAX_LEN_TASK * 3);
        }
        cerr << "Agent#" << index + 1 << " move_plan: " << string(ALL(agent_move_history[index])) << " -> " << s << endl;
        assert (not agent_move_history[index].empty());
        char b = agent_move_history[index].back();
        for (char c : s) {
            auto before_point = get_checkpoint(b);
            auto move_point = get_checkpoint(c);

            // 移動先が同じ場所の場合判定が入らないため別の箇所に移動してからにする
            if (before_point == move_point) {
                int x = move_point.first + 1;
                if (x > AREA_SIZE) {
                    x = move_point.first - 1;
                }
                agent_move_point_queue[index].push({x, move_point.second});
            }

            agent_move_point_queue[index].push(move_point);
            agent_move_history[index].push_back(c);
            while (agent_move_history[index].size() > MAX_LEN_TASK) {
                agent_move_history[index].pop_front();
            }
            b = c;
        }
    }

    int get_now_game_time_ms() {
        auto t = chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - start_time).count();
        return start_game_time_ms + t;
    }

    Move move_next(int index) {
        auto [x, y] = agent_move_point_queue[index].front();
        agent_move_point_queue[index].pop();
        auto move_next_res = call_move_next(index + 1, x, y);

        agent_move_finish_ms[index] = move_next_res.move[1].t + 100;

        // タスクを全てやりきったら次のタスクを取得
        if (agent_move_point_queue[index].empty()) {
            set_move_point(index);
        }

        return move_next_res;
    }

    double get_now_score() {
        double score = 0.0;
        for (const auto& task : game_info.back().task) {
            if (task.total == 0) continue;
            score += (double)(task.weight * task.count) / task.total;
        }

        return score;
    }

    void solve() {
        while (true) {
            int now_game_time_ms = get_now_game_time_ms();

            // エージェントを移動させる
            REP (i, NUM_AGENT) {
                if (agent_move_finish_ms[i] < now_game_time_ms) {
                    auto move_next_res = move_next(i);
                    // 次の移動予定がない場合もう一度実行する
                    if (move_next_res.move.size() == 2) {
                        move_next(i);
                    }
                }
            }

            // ゲーム情報更新
            if (next_call_game_info_time_ms < now_game_time_ms) {
                cerr << "Update GameInfo" << endl;
                game_info.push_back(call_game());
                // 直近 3 分まで保持する
                while (game_info.size() >= 3 * 60 * 1000 / GAME_INFO_SLEEP_TIME) {
                    game_info.pop_front();
                }
                next_call_game_info_time_ms = get_now_game_time_ms() + GAME_INFO_SLEEP_TIME;
                REP (task_index, game_info.back().task.size()) {
                    const Task& task = game_info.back().task[task_index];
                    cerr << "task #" << task_index << ": " << task.s << string(MAX_LEN_TASK - (int)task.s.size(), ' ');
                    cerr << " : predicted score = " << fixed << setprecision(3) << predict_score(task_index);
                    cerr << "; current score = " << fixed << setprecision(3) << calculate_current_score(task);
                    cerr << "; count = " << setw(3) << task.count;
                    cerr << "; total = " << setw(5) << task.total;
                    cerr << endl;
                }
                cerr << "Score: " << get_now_score() << endl;
            }

            this_thread::sleep_for(chrono::milliseconds(10));
        }
    }
};

int main() {
    random_device device;
    mt19937 gen(device());
    Bot bot(gen);
    bot.solve();
    return 0;
}
