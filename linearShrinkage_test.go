// Covariance Shrinkage

package covarianceShrinkage

import (
	"math"
	"testing"
)

func TestCovarianceShrinkage(t *testing.T) {
	type args struct {
		n int
		t int
		X []float64
		S []float64
	}
	tests := []struct {
		name    string
		args    args
		want    []float64
		want1   float64
		wantErr bool
	}{
		{
			name: "mixed returns #1 (n=4,t=3)",
			args: args{
				n: 4,
				t: 3,
				X: []float64{
					4, 1, 3,
					2, 5, 4,
					3, 2, 1,
					1, 6, 7,
				},
			},
			want: []float64{
				2.333333, -2.035832, 0.407353, -3.020793,
				-2.035832, 2.333333, -0.953236, 3.782154,
				0.407353, -0.953236, 1.000000, -2.818354,
				-3.020793, 3.782154, -2.818354, 10.333333,
			},
			want1: 0.8785905137256039,
		},
		{
			name: "mixed returns #2 (n=4,t=3)",
			args: args{
				n: 4,
				t: 3,
				X: []float64{
					1, 2, 3,
					2, 4, 1,
					3, 6, 4,
					4, 8, 2,
				},
			},
			want: []float64{
				1.000000, -0.283402, 0.514419, -0.566805,
				-0.283402, 2.333333, 1.639115, 4.076053,
				0.514419, 1.639115, 2.333333, 3.278231,
				-0.566805, 4.076053, 3.278231, 9.333333,
			},
			want1: 1.2427458025566596,
		},
		{
			name: "perfect multiples (n=5,t=3)",
			args: args{
				n: 5,
				t: 3,
				X: []float64{
					1, 2, 3,
					2, 4, 6,
					3, 6, 9,
					4, 8, 12,
					5, 10, 15,
				},
			},
			want: []float64{
				1, 2, 3, 4, 5,
				2, 4, 6, 8, 10,
				3, 6, 9, 12, 15,
				4, 8, 12, 16, 20,
				5, 10, 15, 20, 25,
			},
			want1: 0.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1, err := CovCor(tt.args.n, tt.args.t, tt.args.X, tt.args.S)
			if (err != nil) != tt.wantErr {
				t.Errorf("CovarianceShrinkage() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			for i := range got {
				if math.Abs(got[i]-tt.want[i]) > 1e-6 {
					t.Errorf("CovarianceShrinkage() element %d: got %v, want %v", i, got[i], tt.want[i])
				}
			}
			if math.Abs(got1-tt.want1) > 1e-6 {
				t.Errorf("CovarianceShrinkage() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}
