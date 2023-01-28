#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <chrono>

using namespace std;
using duration = chrono::microseconds;

int main()
{
    int n;
    cin >> n;

    vector<double> vec(n);
    for (int i = 0; i < n; i++)
    {
        cin >> vec[i];
    }

    chrono::time_point<chrono::high_resolution_clock> start, stop;
    float t = 0;
    start = chrono::high_resolution_clock::now();

    std::reverse(vec.begin(), vec.end());

    stop = chrono::high_resolution_clock::now();
    t += chrono::duration_cast<duration>(stop - start).count();
    cout << "time = " << t << " ms" << endl;

    // for (int i = 0; i < n; i++)
    // {
    //     cout << vec[i] << " ";
    // }
    // cout << endl;

    return 0;
}