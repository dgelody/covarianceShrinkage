// Covariance Shrinkage
//
// Linear shrinkage discussed in http://www.ledoit.net/honey.pdf

package covarianceShrinkage

import (
	"errors"
	"math"

	"github.com/gonum/floats"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Returns a covariance estimate using linear shrinkage.
// The target preserves the diagonal of the sample covariance matrix and
// all correlation coefficients are the same.
//
// Arguments:
// n int: number of assets
// t int: number of periods
// X []float64: matrix of n assets by t periods
// S []float63: nxn covariance matrix
//
// Returns:
// []float64: nxn shrinkage estimator of the covariance matrix
// float64: Frobenius norm between covariance matrix and shrinkage matrix
// error
func CovCor(n int, t int, X []float64, S []float64) (F []float64, norm float64, err error) {
	// Sanity Checks
	if t <= 0 || n <= 0 {
		return nil, 0.0, errors.New("t and n must be > 0")
	}
	if len(X) != t*n {
		return nil, 0.0, errors.New("len(X) != t * n")
	}
	if S != nil && len(S) != n*n {
		return nil, 0.0, errors.New("len(S) != n * n")
	}

	if X == nil {
		return nil, 0.0, errors.New("matrix X is nil")
	}

	var covMatrix []float64
	if S == nil {
		covMatrix = SampleCovarianceMatrix(n, t, X)
	} else {
		covMatrix = S
	}

	//To measure Frobenius norm afterwards
	originalCovMatrix := make([]float64, len(covMatrix))
	copy(originalCovMatrix, covMatrix)

	F = shrinkageTarget(n, covMatrix)
	delta := shrinkageIntensity(n, t, X, covMatrix)

	// shrinkage estimator of the covariance matrix = delta*F + (1-delta)*S
	floats.Scale(delta, F)
	floats.Scale(1.0-delta, covMatrix)
	floats.Add(F, covMatrix)

	froNorm := FrobeniusNorm(originalCovMatrix, F)

	return F, froNorm, nil
}

// Computes the sample covariance matrix
//
// Arguments:
// n int: number of assets
// T int: number of periods
// X []float64: matrix of n assets by t periods
//
// Returns:
// []float64: nxn sample covariance matrix
func SampleCovarianceMatrix(n int, t int, X []float64) []float64 {
	S := make([]float64, n*n)

	// Mean-center each row of X
	means := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < t; j++ {
			sum += X[i*t+j]
		}
		means[i] = sum / float64(t)
	}

	// Deamean the matrix
	tempX := make([]float64, len(X))
	copy(tempX, X)
	for i := 0; i < n; i++ {
		for j := 0; j < t; j++ {
			tempX[i*t+j] -= means[i]
		}
	}

	// Sample covariance matrix S = 1/(T-1) *X_c * (X_c)^T
	// where X_c is the centered matrix
	blas64.Gemm(
		blas.NoTrans, blas.Trans,
		1.0/float64(t-1),
		blas64.General{Rows: n, Cols: t, Stride: t, Data: tempX},
		blas64.General{Rows: n, Cols: t, Stride: t, Data: tempX},
		0.0,
		blas64.General{Rows: n, Cols: n, Stride: n, Data: S},
	)

	return S
}

// Computes the Frobenius norm between matrices A and B
//
// Arguments:
// A []float64: matrix A
// B []float64: matrix B
//
// Returns:
// float64: Frobenius norm
func FrobeniusNorm(A, B []float64) float64 {
	if len(A) != len(B) {
		panic("matrices must be the same size")
	}
	var sum float64
	for i := range A {
		diff := A[i] - B[i]
		sum += diff * diff
	}

	// fmt.Println(A)
	// fmt.Println(B)
	// fmt.Println(sum)

	return math.Sqrt(sum)
}

// Returns the constant correlation model.
//
// Arguments:
// n int: number of assets
// s []float64: nxn sample covariance matrix
//
// Returns:
// F []float64: nxn shrinkage target
func shrinkageTarget(n int, s []float64) (F []float64) {
	r_bar := avgSampleCorr(n, s)

	F = make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				F[i*n+j] = s[i*n+i]
			} else {
				F[i*n+j] = r_bar * math.Sqrt(s[i*n+i]*s[j*n+j])
			}
		}
	}

	return F
}

// Returns the 'optimal' shrinkage constant
// Note: It is the one that minimizes the expected distance between the shrinkage
// estimator and the true covariance matrix
//
// Arguments:
// n int: number of assets
// T int: number of periods
// X []float64: matrix of n assets by t periods
// s []float64: nxn sample covariance matrix
//
// Returns:
// delta float64: 'optimal' shrinkage constant
func shrinkageIntensity(n int, T int, X []float64, s []float64) (delta float64) {

	// Unbiased estimator factor
	T1 := float64(T - 1)
	// compute mean of each row
	means := make([]float64, n)
	for i := 0; i < n; i++ {
		means[i] = sumRow(X, i, T) / float64(T)
	}

	// Estimator for pi
	pi := make([]float64, n*n)
	pi_hat := 0.0
	for i := 0; i < n; i++ {
		X_i_bar := means[i]
		for j := 0; j < n; j++ {
			X_j_bar := means[j]
			pi_ij := 0.0
			for t := 0; t < T; t++ {
				X_it := X[i*T+t] - X_i_bar
				X_jt := X[j*T+t] - X_j_bar
				pi_ij += X_it * X_it * X_jt * X_jt
			}
			pi_ij /= T1
			pi_ij -= s[i*n+j] * s[i*n+j]

			pi_hat += pi_ij
			pi[i*n+j] = pi_ij
		}
	}

	// Estimator for rho
	rho_hat := 0.0
	r_bar := avgSampleCorr(n, s)

	for i := 0; i < n; i++ {
		X_i_bar := means[i]
		for j := 0; j < n; j++ {
			X_j_bar := means[j]

			if i == j {
				rho_hat += pi[i*n+i]
				continue
			}

			rho_contrib := 0.0
			for t := 0; t < T; t++ {
				X_it := X[i*T+t] - X_i_bar
				X_jt := X[j*T+t] - X_j_bar
				rho_contrib += X_it * X_it * X_it * X_jt
			}

			rho_contrib /= T1
			rho_contrib -= s[i*n+i] * s[i*n+j]
			rho_contrib *= r_bar * math.Sqrt(s[j*n+j]/s[i*n+i])
			rho_hat += rho_contrib
		}
	}

	// Estimator for gamma
	gamma_hat := 0.0
	F := shrinkageTarget(n, s)
	for i := 0; i < n*n; i++ {
		diff := F[i] - s[i]
		gamma_hat += diff * diff
	}

	// Estimator for kappa
	kappa_hat := (pi_hat - rho_hat) / gamma_hat

	// Although very unlikely, in principle it can happen in finite sample that κ/T < 0 or that κ/T > 1,
	// in which case we simply truncate it at 0 or at 1, respectively.
	delta = math.Max(0.0, math.Min(kappa_hat/T1, 1.0))

	if gamma_hat == 0.0 {
		delta = 1
	}

	// fmt.Println("pi hat: ", pi_hat)
	// fmt.Println("rho hat: ", rho_hat)
	// fmt.Println("gamma hat: ", gamma_hat)

	// fmt.Println("Kappa_hat: ", kappa_hat)
	// fmt.Println("t1: ", T1)

	return delta
}

// Computes the sum of the entires in 'row' in the matrix data
func sumRow(data []float64, row, cols int) float64 {
	sum := 0.0
	start := row * cols
	for i := 0; i < cols; i++ {
		sum += data[start+i]
	}
	return sum
}

// Computes average off-diagonal sample correlation
func avgSampleCorr(n int, S []float64) float64 {
	rSum := 0.0
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			rSum += S[i*n+j] / math.Sqrt(S[i*n+i]*S[j*n+j])
		}
	}
	return 2.0 * rSum / (float64(n) * float64(n-1))
}
