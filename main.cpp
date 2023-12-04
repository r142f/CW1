#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <omp.h>

const int NUM_THREADS = 4;
const int ARRAY_SIZE = 1e8;
const int BLOCK = 1000;
const int RUNS = 5;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<> dis(0, ARRAY_SIZE);

int partition(std::vector<int> &a, int l, int r) {
    int v = a[(l + r) / 2];
    int i = l, j = r;
    while (i <= j) {
        while (a[i] < v) {
            i++;
        }
        while (a[j] > v) {
            j--;
        }
        if (i >= j) {
            break;
        }
        std::swap(a[i], a[j]);
        i++;
        j--;
    }

    return j;
}

void SequentialQuickSort(std::vector<int> &a, int l, int r) {
    if (l < r) {
        int q = partition(a, l, r);
        SequentialQuickSort(a, l, q - 1);
        SequentialQuickSort(a, q + 1, r);
    }
}

void SubParallelQuickSort(std::vector<int> &a, int l, int r) {
    if (r - l < BLOCK) {
        return SequentialQuickSort(a, l, r);
    } else {
        int q = partition(a, l, r);
#pragma omp task shared(a)
        SubParallelQuickSort(a, l, q - 1);
#pragma omp task shared(a)
        SubParallelQuickSort(a, q + 1, r);
    }
}

void ParallelQuickSort(std::vector<int> &a, int l, int r) {
#pragma omp parallel
    {
#pragma omp single
        SubParallelQuickSort(a, l, r);
#pragma omp taskwait
    }
}

void Experiment() {
    long parallel_duration = 0;
    long sequential_duration = 0;
    for (int run = 1; run <= RUNS; run++) {
        std::vector<int> a(ARRAY_SIZE);
        for (int i = 0; i < ARRAY_SIZE; i++) {
            a[i] = dis(gen);
        }
        std::vector<int> b(a);

        auto start = std::chrono::high_resolution_clock::now();
        ParallelQuickSort(a, 0, a.size() - 1);
        auto finish = std::chrono::high_resolution_clock::now();


        std::cout << "Run #" << run << ": par took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << "ms" << std::endl;
        parallel_duration += std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();

        start = std::chrono::high_resolution_clock::now();
        SequentialQuickSort(b, 0, b.size() - 1);
        finish = std::chrono::high_resolution_clock::now();
        std::cout << "Run #" << run << ": seq took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() << "ms" << std::endl;
        sequential_duration += std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
    }

    std::cout << "Average parallel result: " << double(parallel_duration) / double(RUNS) << "ms\n";
    std::cout << "Average sequential result: " << double(sequential_duration) / double(RUNS) << "ms\n";
}

int main() {
    omp_set_num_threads(NUM_THREADS);

    Experiment();

    return 0;
}
