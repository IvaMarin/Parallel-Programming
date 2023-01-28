// g++ -std=c++11 -O2 -Wextra -Wall -Wno-sign-compare -Wno-unused-result -o make_tests make_tests.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <map>
#include <tuple>

using namespace std;

int main()
{
    // vector<int> tests = {70, 223, 707, 2236, 7071, 20000};
    srand(time(NULL));
    // for (auto t : tests)
    // {
    // string s = to_string(t * t);
    string s = "silent_hill_2_remake";
    string in_file_name = "in/" + s + ".data";
    string out_file_name = "out/" + s + ".data";
    // int nc = 2 + rand() % 32;
    int nc = 7;

    // ofstream data_file(in_file_name, ios::out | ios::binary);
    int w = 1920, h = 1080;
    // if (data_file.is_open())
    // {
    //     data_file.write((char *)&w, sizeof(w));
    //     data_file.write((char *)&h, sizeof(h));
    //     int d = 0x00030201;
    //     for (int i = 0; i < w * h; i++)
    //     {
    //         data_file.write((char *)&d, sizeof(d));
    //     }
    //     data_file.close();
    // }
    // else
    // {
    //     return 1;
    // }

    ofstream test_file;
    string test_file_name = "tests/test_" + s + ".t";
    test_file.open(test_file_name);
    test_file << in_file_name << endl;
    test_file << out_file_name << endl;
    test_file << nc << endl;
    vector<vector<int>> img(w, vector<int>(h));
    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            img[i][j] = 0 + rand() % nc;
        }
    }
    vector<vector<pair<int, int>>> classes(nc);
    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            classes[img[i][j]].push_back({i, j});
        }
    }
    int np;
    for (int i = 0; i < nc; i++)
    {
        np = classes[i].size();
        test_file << np << " ";
        for (size_t j = 0; j < np; j++)
        {
            test_file << classes[i][j].first << " " << classes[i][j].second << " ";
        }
        test_file << endl;
    }

    test_file.close();
    // }
    return 0;
}