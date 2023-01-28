// g++ -std=c++11 -O2 -Wextra -Wall -Wno-sign-compare -Wno-unused-result -o make_tests make_tests.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <random>

using namespace std;

int main()
{
    vector<int> tests = {70, 223, 707, 2236, 7071};
    srand(time(NULL));
    for (auto t : tests)
    {
        string s = to_string(t * t);
        string in_file_name = "in/in_" + s + ".data";
        string out_file_name = "out/out_" + s + ".data";
        int r = 0 + rand() % 1024;

        ofstream data_file(in_file_name, ios::out | ios::binary);
        int w = t, h = t;
        if (data_file.is_open())
        {
            data_file.write((char *)&w, sizeof(w));
            data_file.write((char *)&h, sizeof(h));
            int d = 0x00030201;
            for (int i = 0; i < w * h; i++)
            {
                data_file.write((char *)&d, sizeof(d));
            }
            data_file.close();
        }
        else
        {
            return 1;
        }

        ofstream test_file;
        string test_file_name = "tests/test_" + s + ".t";
        test_file.open(test_file_name);
        test_file << in_file_name << endl;
        test_file << out_file_name << endl;
        test_file << r;
        test_file.close();
    }
    return 0;
}