// Covariance Shrinkage

package covarianceShrinkage

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestQuadraticInverseShrinkage(t *testing.T) {
	type args struct {
		X          *mat.Dense
		demeanData bool
	}
	tests := []struct {
		name    string
		args    args
		want    []float64
		want1   float64
		wantErr bool
	}{
		{
			name: "mixed returns #1 (n=3,t=5)",
			args: args{
				demeanData: true,
				X: mat.NewDense(5, 3, []float64{
					0.189053, -0.522748, -0.413064,
					-2.441467, 1.799707, 1.144166,
					-0.325423, 0.773807, 0.281211,
					-0.553823, 0.977567, -0.310557,
					-0.328824, -0.792147, 0.454958,
				}),
			},
			want: []float64{
				0.968403, -0.657597, -0.471565,
				-0.657597, 1.135013, 0.193892,
				-0.471565, 0.193892, 0.499316,
			},
			want1: 0.6575965297355398,
		},
		{
			name: "mixed returns #2 (n=5,t=3)",
			args: args{
				demeanData: true,
				X: mat.NewDense(3, 5, []float64{
					0.189053, -0.522748, -0.413064, -2.441467, 1.799707,
					1.144166, -0.325423, 0.773807, 0.281211, -0.553823,
					0.977567, -0.310557, -0.328824, -0.792147, 0.454958,
				}),
			},
			want: []float64{
				0.517310, 0.009488, 0.262506, 0.351251, -0.317579,
				0.009488, 0.428719, 0.075625, 0.069702, -0.066022,
				0.262506, 0.075625, 0.367747, 0.488561, -0.387184,
				0.351251, 0.069702, 0.488561, 1.466359, -0.900320,
				-0.317579, -0.066022, -0.387184, -0.900320, 1.207849,
			},
			want1: 0.900320113449109,
		},
		{
			name: "scalar asset (n=1,t=10)",
			args: args{
				demeanData: true,
				X: mat.NewDense(10, 1, []float64{
					0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
				}),
			},
			want:  []float64{9.166667},
			want1: 0.0,
		},
		{
			name: "perfect collinearity (dup column)",
			args: args{
				demeanData: true,
				X: mat.NewDense(6, 3, []float64{
					1, 1, 3,
					2, 2, 1,
					3, 3, 4,
					4, 4, 1,
					5, 5, 5,
					6, 6, 9,
				}),
			},
			want: []float64{
				3.620552, 3.620552, 2.943759,
				3.620552, 3.620552, 2.943759,
				2.943759, 2.943759, 8.725563,
			},
			want1: 3.6205518094671736,
		},
		{
			name: "zero variance asset",
			args: args{
				demeanData: true,
				X: mat.NewDense(8, 4, []float64{
					0.1, -0.2, 0.3, 0,
					-0.4, 0.5, -0.6, 0,
					0.7, -0.8, 0.9, 0,
					-1.0, 1.1, -1.2, 0,
					-1.3, 1.4, -1.5, 0,
					1.6, -1.7, 1.8, 0,
					-1.9, 2.0, -2.1, 0,
					2.2, -2.3, 2.4, 0,
				}),
			},
			want: []float64{
				2.051885, -2.182321, 2.312756, 0.0,
				-2.182321, 2.325103, -2.467884, 0.0,
				2.312756, -2.467884, 2.623012, 0.0,
				0.0, 0.0, 0.0, 0.0,
			},
			want1: 2.4678843272542093,
		},
		{
			name: "all equal values (rank-1)",
			args: args{
				demeanData: true,
				X: mat.NewDense(12, 3, []float64{
					7, 7, 7,
					7, 7, 7,
					7, 7, 7,
					7, 7, 7,
					7, 7, 7,
					7, 7, 7,
					7, 7, 7,
					7, 7, 7,
					7, 7, 7,
					7, 7, 7,
					7, 7, 7,
					7, 7, 7,
				}),
			},
			want: []float64{
				0.0, 0.0, 0.0,
				0.0, 0.0, 0.0,
				0.0, 0.0, 0.0,
			},
			want1: 1.7320508075688772e-12,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1, err := QuadraticInverseShrinkage(tt.args.X, nil, tt.args.demeanData)
			if (err != nil) != tt.wantErr {
				t.Errorf("QuadraticInverseShrinkage() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			gotData := got.RawMatrix().Data

			for i := range gotData {
				if math.Abs(gotData[i]-tt.want[i]) > 1e-6 {
					t.Errorf("QuadraticInverseShrinkage() element %d: got %v, want %v", i, gotData[i], tt.want[i])
				}
			}
			if math.Abs(got1-tt.want1) > 1e-6 {
				t.Errorf("QuadraticInverseShrinkage() got1 = %v, want1 %v", got1, tt.want1)
			}
		})
	}
}
