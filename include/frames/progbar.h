#include <iostream>
#include <chrono>
#include <ratio>

namespace utility
{
class progbar
{
private:
    int call_count = 0;

    // formatting:
    std::ostream& out;
    int width;
    char fill;
    char space;

    // timing:
    using clock = std::chrono::steady_clock;  // low-precision but monotone
    bool show_time;
    clock::time_point start_time;

    inline void write_time(std::ostream& out, std::chrono::duration<float> dur)
    {
        using namespace std::chrono_literals;
        using namespace std::chrono;
        using std::milli;
        using std::ratio;

        // save/restore stream's precision setting
        auto old_prec = out.precision();

        out.precision(3);
        if (dur > 1h)
        {
            out << duration<float, ratio<3600>>(dur).count() << "h";
        }
        else if (dur > 1min)
        {
            out << duration<float, ratio<60>>(dur).count() << "m";
        }
        else if (dur > 1s)
        {
            out << duration<float>(dur).count() << "s";
        }
        else
        {
            out << duration<float, milli>(dur).count() << "ms";
        }
        out.precision(old_prec);
    }

public:
    progbar(std::ostream& out_, int width_ = 40, char fill_ = '#', char space_ = '.', bool show_time_ = true)
        : out(out_), width(width_), fill(fill_), space(space_), show_time(show_time_)
    {
    }

    void update(uint64_t cur, uint64_t tot)
    {
        // up one line, clear current line, and beginning of line (CR)
        out << "\033[A\033[2K\r";

        // output bar:
        int fills = (int)((float)cur / tot * width);
        out << '[';
        for (int i = 0; i < width; i++)
        {
            out << (i < fills ? fill : space);
        }
        out << "] " << cur << '/' << tot;

        // time estimate
        if (show_time)
        {
            if (call_count == 0)
            {
                start_time = clock::now();
            }
            else if (cur > 0 && cur < tot)
            {
                // if not first call, output time estimate for remaining time
                out << ' ';
                auto elapsed = (clock::now() - start_time);
                auto estimate = elapsed / cur * (tot - cur);
                write_time(out, estimate);
                out << " remaining; ";
                write_time(out, elapsed);
                out << " elapsed.";
            }
            else
            {
                out << " done";
            }
        }

        out << std::endl;
        call_count++;
    }

};  // class progbar
}  // namespace utility
