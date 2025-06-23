# covarianceShrinkage

A Go package implementing Ledoit-Wolf linear and nonlinear shrinkage for improving the estimation of large covariance matrices. Particularly useful in high-dimensional statistics and portfolio optimization, where the number of assets exceeds the number of observations.

---

## Installation

```sh
go get github.com/dgelody/covarianceShrinkage
```

---

## Functions

### `CovCor`

**Signature:**

```go
func CovCor(n int, t int, X []float64, S []float64) (F []float64, norm float64, err error) 
```

**Description:**

Linear shrinkage towards constant-correlation matrix; the target preserves the diagonal of the sample covariance matrix and all correlation coefficients are the same. See Ledoit and Wolf (2004a).

**Parameters:**

- `X` - matrix of `n` by `t` observations. The data used to create the covariance matrix. Should be differenced if the covariance matrix is the covariance of the differences.
- `S` - `n` by `n` covariance matrix. If nil, `S` will be the sample covariance matrix

**Returns:**

- `F` - `n` by `n` shrinkage estimator of the covariance matrix
- `norm` - Frobenius norm between the covariance matrix `S` and the shrinkage estimator of the covariance matrix `F`
- `err` - error

### `QuadraticInverseShrinkage`

**Signature:**

```go
func QuadraticInverseShrinkage(X []float64, t, n, k int) (F []float64, norm float64, err error)
```

**Description:**

Nonlinear shrinkage derived under Frobenius loss and its two cousins, Inverse Stein’s loss and Minimum Variance loss, called quadratic-inverse shrinkage (QIS). Preserves the variance of the original covariance matrix. See Ledoit and Wolf (2022, Section 4.5).

**Parameters:**

- `X` - matrix of t observations by n assets
- `t` — number of observations
- `n` — number of random variables
- `k` —  if `k < 0`,then the algorithm demeans the data by default, and adjusts the effective sample size accordingly. If the user inputs `k == 0`, then no demeaning takes place; if user inputs `k == 1`, then it signifies that the data Y have already been demeaned.

**Returns:**

- `F` - `n` by `n` shrinkage estimator of the covariance matrix
- `norm` - Frobenius norm between the covariance matrix `S` and the shrinkage estimator of the covariance matrix `F`
- `err` - error

---

## Testing

Run all unit tests with coverage:

```sh
go test -v -cover ./...
```

---

## References

(a) Ledoit, O. and Wolf, M. (2004a). Honey, I shrunk the sample covariance matrix. Journal of Portfolio Management, 30(4):110–119.

(b) Ledoit, O. and Wolf, M. (2004b). A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis, 88(2):365–411.

(c) Ledoit, O. and Wolf, M. (2022). Quadratic shrinkage for large covariance matrices. Bernoulli, 28(3): 1519-1547.

---

## License

MIT License. See the `LICENSE` file for details.
