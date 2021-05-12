import multiprocessing 
from joblib import Parallel, delayed


def sum(a,b):
	return a+b



if __name__ == "__main__":
	print(f"Number of cpu : {multiprocessing.cpu_count()}")

	res = Parallel(n_jobs=4)(delayed(sum)(i,i)for i in range(10000000))
	print(res)

