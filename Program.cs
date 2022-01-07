using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;
using System.Collections;

namespace NameSort
{
    // Create a name class to get the first and last name from the parameter of the list
    public class name : IComparable<name>
    {
        public name(string fname, string lname)
        {
            this.firstName = fname;
            this.lastName = lname;
        }

        public string firstName { get; set; }
        public string lastName { get; set; }
        //public int Length { get; set; }

        public int CompareTo(name b)
        {
            if (string.Compare(this.lastName, b.lastName) == 0 && string.Compare(this.firstName, b.firstName) == 1)
            {
                return 1;
            }

            else if (string.Compare(this.lastName, b.lastName) == 1)
            {
                return 1;
            }

            else
            {
                return 0;
            }
        }
    }


    /// 

    public class ParallelSort
    {
        #region Public Static Methods

        /// <summary>
        /// Sequential quicksort.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="arr"></param>
        public static void QuicksortSequential(List<name> arr)
        {
            QuicksortSequential(arr, 0, arr.Count - 1);
        }

        /// <summary>
        /// Parallel quicksort
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="arr"></param>
        public static void QuicksortParallel(List<name> arr)
        {
            QuicksortParallel(arr, 0, arr.Count - 1);
        }

        #endregion

        #region Private Static Methods

        private static void QuicksortSequential(List<name> arr, int left, int right)
        {
            if (right > left)
            {
                int pivot = Partition(arr, left, right);
                QuicksortSequential(arr, left, pivot - 1);
                QuicksortSequential(arr, pivot + 1, right);
            }
        }

        private static void QuicksortParallel(List<name> arr, int left, int right)

        {
            const int SEQUENTIAL_THRESHOLD = 2048;
            if (right > left)
            {
                if (right - left < SEQUENTIAL_THRESHOLD)
                {
                    QuicksortSequential(arr, left, right);
                }
                else
                {
                    int pivot = Partition(arr, left, right);
                    Parallel.Invoke(new Action[] { delegate {QuicksortParallel(arr, left, pivot - 1);},
                                               delegate {QuicksortParallel(arr, pivot + 1, right);}
                });
                }
            }
        }

        private static void Swap(List<name> arr, int i, int j)
        {
            name tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }

        private static int Partition(List<name> arr, int low, int high)
        {
            // Simple partitioning implementation
            int pivotPos = (high + low) / 2;
            name pivot = arr[pivotPos];
            Swap(arr, low, pivotPos);

            int left = low;
            for (int i = low + 1; i <= high; i++)
            {
                if (arr[i].CompareTo(pivot) < 1)
                {
                    left++;
                    Swap(arr, i, left);
                }
            }

            Swap(arr, low, left);
            return left;
        }

        #endregion
    }
    class Program
    {
        // Main entry
        static void Main(string[] args)
        {
            // Define an empty list to allocate the name
            List<name> Names = new List<name>();

            // populate the list of names from a file
            using (StreamReader sr = new StreamReader("names.txt"))
            {
                // read each name from the list
                while (sr.Peek() >= 0)
                {
                    // split the first and last name by blank
                    string[] s = sr.ReadLine().Split(' ');
                    // append the each first and last name into a list
                    Names.Add(new name(s[0], s[1]));
                }
            }

            Console.WriteLine("Sorting...");
            // time the sort.
            //List<name> sortedNames = ParallelSort.QuicksortSequential(T[] Names, 0, Names.Count);


            Stopwatch stopwatch = Stopwatch.StartNew();

            //ParallelSort sortednames = new ParallelSort();

            //List<name> sortedNames2 = 
            ParallelSort.QuicksortParallel(Names);

            //List<name> sortedNames = Names.OrderBy(s => s.lastName).ThenBy(s => s.firstName).ToList();
            stopwatch.Stop();
            Console.WriteLine("Code took {0} milliseconds to execute parralel", stopwatch.ElapsedMilliseconds);

            Stopwatch stopwatch2 = Stopwatch.StartNew();

            //ParallelSort sortednames = new ParallelSort();

            //List<name> sortedNames2 = 
            //ParallelSort.QuicksortParallel(Names);

            List<name> sortedNames = Names.OrderBy(s => s.lastName).ThenBy(s => s.firstName).ToList();
            stopwatch2.Stop();
            Console.WriteLine("Code took {0} milliseconds to execute sequential", stopwatch2.ElapsedMilliseconds);

            Console.ReadLine();
        }
    }
}
