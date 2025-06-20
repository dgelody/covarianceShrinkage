// Covariance Shrinkage

package covarianceShrinkage

import (
	"math"
	"testing"
)

func TestQuadraticInverseShrinkage(t *testing.T) {
	type args struct {
		X []float64
		t int
		n int
		k int
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
				t: 5, n: 3, k: -1,
				X: []float64{
					0.189053, -0.522748, -0.413064, // row 0
					-2.441467, 1.799707, 1.144166, // row 1
					-0.325423, 0.773807, 0.281211, // row 2
					-0.553823, 0.977567, -0.310557, // row 3
					-0.328824, -0.792147, 0.454958, // row 4
				},
			},
			want: []float64{
				0.968403, -0.657597, -0.471565,
				-0.657597, 1.135013, 0.193892,
				-0.471565, 0.193892, 0.499316,
			},
			want1: 0.38625046149603826,
		},
		{
			name: "mixed returns #2 (n=5,t=3)",
			args: args{
				t: 3, n: 5, k: -1,
				X: []float64{
					0.189053, -0.522748, -0.413064, -2.441467, 1.799707, // row 0
					1.144166, -0.325423, 0.773807, 0.281211, -0.553823, // row 1
					0.977567, -0.310557, -0.328824, -0.792147, 0.454958, // row 2
				},
			},
			want: []float64{
				0.517310, 0.009488, 0.262506, 0.351251, -0.317579,
				0.009488, 0.428719, 0.075625, 0.069702, -0.066022,
				0.262506, 0.075625, 0.367747, 0.488561, -0.387184,
				0.351251, 0.069702, 0.488561, 1.466359, -0.900320,
				-0.317579, -0.066022, -0.387184, -0.900320, 1.207849,
			},
			want1: 1.4719257264587164,
		},

		{
			name: "scalar asset (n=1,t=10)",
			args: args{
				t: 10, n: 1, k: -1,
				X: []float64{
					0, 1, 2, 3, 4, 5, 6, 7, 8, 9, // each is a row
				},
			},
			want:  []float64{9.166667},
			want1: 0.0,
		},
		{
			name: "perfect collinearity (dup column)",
			args: args{
				t: 6, n: 3, k: -1,
				// col0 == col1, col2 independent
				X: []float64{
					1, 1, 3,
					2, 2, 1,
					3, 3, 4,
					4, 4, 1,
					5, 5, 5,
					6, 6, 9,
				},
			},
			want: []float64{3.620552, 3.620552, 2.943759,
				3.620552, 3.620552, 2.943759,
				2.943759, 2.943759, 8.725563},
			want1: 1.9426407076621128,
		},
		{
			name: "zero variance asset",
			args: args{
				t: 8, n: 4, k: -1,
				// last asset is all zeros
				X: []float64{
					0.1, -0.2, 0.3, 0,
					-0.4, 0.5, -0.6, 0,
					0.7, -0.8, 0.9, 0,
					-1.0, 1.1, -1.2, 0,
					-1.3, 1.4, -1.5, 0,
					1.6, -1.7, 1.8, 0,
					-1.9, 2.0, -2.1, 0,
					2.2, -2.3, 2.4, 0,
				},
			},
			want: []float64{2.051885, -2.182321, 2.312756, 0.0,
				-2.182321, 2.325103, -2.467884, 0.0,
				2.312756, -2.467884, 2.623012, 0.0,
				0.0, 0.0, 0.0, 0.0},
			want1: 0.0026083582107935045,
		},
		{
			name: "all equal values (rank-1)",
			args: args{
				t: 12, n: 3, k: -1,
				X: []float64{
					7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
					7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
					7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
				},
			},
			want:  []float64{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
			want1: 1.7320508075688772e-12,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1, err := QuadraticInverseShrinkage(tt.args.X, tt.args.t, tt.args.n, tt.args.k)
			if (err != nil) != tt.wantErr {
				t.Errorf("QuadraticInverseShrinkage() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			for i := range got {
				if math.Abs(got[i]-tt.want[i]) > 1e-6 {
					t.Errorf("QuadraticInverseShrinkage() element %d: got %v, want %v", i, got[i], tt.want[i])
				}
			}
			if math.Abs(got1-tt.want1) > 1e-6 {
				t.Errorf("QuadraticInverseShrinkage() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}
