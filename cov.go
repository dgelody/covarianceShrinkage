// Covariance Shrinkage
//
// Linear shrinkage discussed in http://www.ledoit.net/honey.pdf
// Quadratic Inverse shrinkage discussed in http://www.ledoit.net/BEJ1911-021R1A0.pdf

package covarianceShrinkage

import (
	"errors"
	"fmt"
	"math"

	"github.com/gonum/floats"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
)

// Returns a covariance estimate using linear shrinkage.
// The target preserves the diagonal of the sample covariance matrix and
// all correlation coefficients are the same.
//
// Arguments:
// n int: number of assets
// t int: number of periods
// X []float64: matrix of n assets by t periods
//
// Returns:
// []float64: nxn shrinkage estimator of the covariance matrix
func CovarianceShrinkage(n int, t int, X []float64) ([]float64, error) {
	// Sanity Checks
	if t <= 0 || n <= 0 {
		return nil, errors.New("t and n must be > 0")
	}
	if len(X) != t*n {
		return nil, errors.New("len(X) != t * n")
	}

	S := sampleCovarianceMatrix(n, t, X)
	F := shrinkageTarget(n, S)
	delta := shrinkageIntensity(n, t, X, S)

	// shrinkage estimator of the covariance matrix = delta*F + (1-delta)*S

	floats.Scale(delta, F)
	floats.Scale(1.0-delta, S)
	floats.Add(F, S)

	return F, nil
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

// Computes the sample covariance matrix
func sampleCovarianceMatrix(n int, t int, X []float64) []float64 {
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

// Returns the Quadratic-Inverse-Shrinkage covariance estimator.
//
// Arguments:
// X []float64: matrix of t periods by n assets
// t int: number of observations (time periods)
// n int: number of assets (variables)
// k int: lag parameter (use 1 for standard “demean, divide by t-1”)
//
// Returns:
// []float64: the Quadratic-Inverse-Shrinkage covariance estimator
// error
func QuadraticInverseShrinkage(X []float64, t, n, k int) ([]float64, error) {
	// Sanity Checks
	if t <= 0 || n <= 0 {
		return nil, errors.New("QIS: t and n must be > 0")
	}
	if len(X) != t*n {
		return nil, errors.New("QIS: len(X) != t * n")
	}
	if k <= 0 {
		k = 1
	}

	Y := mat.NewDense(t, n, X)

	// Demean the matrix X
	tEff := t - k
	if tEff <= 0 {
		return nil, errors.New("QIS: effective sample size t-k ≤ 0")
	}
	Yc := mat.DenseCopyOf(Y)
	for j := 0; j < n; j++ {
		colMean := mat.Sum(Y.ColView(j)) / float64(t)
		for i := 0; i < t; i++ {
			Yc.Set(i, j, Yc.At(i, j)-colMean)
		}
	}

	// Compute sample covariance matrix S
	var S mat.Dense
	S.Mul(Yc.T(), Yc)
	S.Scale(1/float64(tEff), &S)

	// Enforce symmetry numerically
	var tmp mat.Dense
	tmp.Add(&S, S.T())
	tmp.Scale(0.5, &tmp)
	S.CloneFrom(&tmp)

	// Eigen-decomposition
	var eig mat.EigenSym
	Ssym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			Ssym.SetSym(i, j, S.At(i, j))
		}
	}
	if ok := eig.Factorize(Ssym, true); !ok {
		return nil, errors.New("QIS: eigendecomposition failed")
	}

	lambda := eig.Values(nil)
	fmt.Println("The eigenvalues of the matrix are:")
	fmt.Println(lambda)
	const eps = 1e-12
	for i := range lambda {
		if lambda[i] < eps {
			lambda[i] = eps // Reset negative values to 0
		}
	}

	var U mat.Dense
	eig.VectorsTo(&U) // The columns of U correspond to eigenvalues in monotone order

	// Constants
	tF := float64(tEff)
	nF := float64(n)
	c := nF / tF
	h := math.Pow(math.Min(c*c, 1/(c*c)), 0.35) / math.Pow(nF, 0.35) // Smoothing parameter
	m := int(math.Min(nF, tF))                                       //number of non-null eigenvalues
	start := n - m
	invLambda := make([]float64, m) //inverse of (non-null) eigenvalues
	for i := 0; i < m; i++ {
		invLambda[i] = 1 / lambda[start+i]
	}

	// Compute Quadratic-Inverse Shrinkage estimator of the covariance matrix
	theta := make([]float64, m)  //smoothed stein shrinker
	Htheta := make([]float64, m) //its conjugate
	for j := 0; j < m; j++ {
		invLJ := invLambda[j]
		var sumT, sumH float64
		for i := 0; i < m; i++ {
			invLI := invLambda[i]
			diff := invLI - invLJ
			den := diff*diff + invLI*invLI*h*h
			sumT += invLI * diff / den
			sumH += invLI * invLI * h / den
		}
		theta[j] = sumT / float64(m)
		Htheta[j] = sumH / float64(m)
	}

	Atheta2 := make([]float64, m) //its squared amplitude
	for i := range Atheta2 {
		Atheta2[i] = theta[i]*theta[i] + Htheta[i]*Htheta[i]
	}

	fmt.Println("h =", h)
	fmt.Println("m =", m, " start =", start)
	fmt.Println("invLambda =", invLambda)
	fmt.Println("theta     =", theta)
	fmt.Println("Htheta    =", Htheta)
	fmt.Println("Atheta²   =", Atheta2)

	// Shrink eigenvalues
	delta := make([]float64, n)
	if n <= tEff {
		for j := 0; j < m; j++ {
			invLJ := invLambda[j]
			tj := theta[j]
			aj := Atheta2[j]
			den := (1-c)*(1-c)*invLJ + 2*c*(1-c)*invLJ*tj + c*c*invLJ*aj
			delta[start+j] = 1 / den
		}
	} else {
		meanInv := floats.Sum(invLambda) / float64(m)
		delta0 := 1 / ((c - 1) * meanInv)
		for i := 0; i < n-(tEff); i++ {
			delta[i] = delta0
		}
		for j := 0; j < m; j++ {
			delta[start+j] = 1 / (invLambda[j] * Atheta2[j])
		}
	}

	// Preserve Trace
	scale := floats.Sum(lambda) / floats.Sum(delta)

	fmt.Printf("scale factor      = %.12g\n", scale)
	fmt.Printf("trace(delta)      = %.12g\n", floats.Sum(delta))
	fmt.Printf("trace(sample Σ)   = %.12g\n", floats.Sum(lambda))

	floats.Scale(scale, delta)

	fmt.Printf("trace(QIS Σ̂)     = %.12g\n", floats.Sum(delta)) // should match line above

	// Compute qis covariance estimator
	fmt.Printf("U (eigenvectors):\n%v\n\n",
		mat.Formatted(&U, mat.Prefix(" "), mat.Squeeze()))

	diagDelta := mat.NewDiagDense(n, delta)
	var tmp2 mat.Dense
	tmp2.Mul(&U, diagDelta)
	var sigma mat.Dense
	sigma.Mul(&tmp2, U.T())

	fmt.Printf("sigma:\n%v\n\n",
		mat.Formatted(&sigma, mat.Prefix(" "), mat.Squeeze()))

	rm := sigma.RawMatrix()
	out := make([]float64, len(rm.Data))
	copy(out, rm.Data)
	return out, nil
}
