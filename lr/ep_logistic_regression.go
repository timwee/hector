package lr

import (
	"bufio"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/util"
	"math"
	"os"
	"strconv"
	"strings"
)

type EPLogisticRegressionParams struct {
	init_var, beta float64
}

type EPLogisticRegression struct {
	Model  map[int64]*util.Gaussian
	params EPLogisticRegressionParams
}

func (algo *EPLogisticRegression) SaveModel(path string) {
	sb := util.StringBuilder{}
	for f, g := range algo.Model {
		sb.Int64(f)
		sb.Write("\t")
		sb.Float(g.Mean)
		sb.Write("\t")
		sb.Float(g.Vari)
		sb.Write("\n")
	}
	sb.WriteToFile(path)
}

func (algo *EPLogisticRegression) LoadModel(path string) {
	file, _ := os.Open(path)
	defer file.Close()

	scaner := bufio.NewScanner(file)
	for scaner.Scan() {
		line := scaner.Text()
		tks := strings.Split(line, "\t")
		fid, _ := strconv.ParseInt(tks[0], 10, 64)
		mean, _ := strconv.ParseFloat(tks[1], 64)
		vari, _ := strconv.ParseFloat(tks[2], 64)
		g := util.Gaussian{Mean: mean, Vari: vari}
		algo.Model[fid] = &g
	}
}

func (algo *EPLogisticRegression) Predict(sample *core.Sample) float64 {
	s := util.Gaussian{Mean: 0.0, Vari: 0.0}
	for _, feature := range sample.Features {
		if feature.Value == 0.0 {
			continue
		}
		wi, ok := algo.Model[feature.Id]
		if !ok {
			wi = &(util.Gaussian{Mean: 0.0, Vari: algo.params.init_var})
		}
		s.Mean += feature.Value * wi.Mean
		s.Vari += feature.Value * feature.Value * wi.Vari
	}

	t := s
	t.Vari += algo.params.beta
	return t.Integral(t.Mean / math.Sqrt(t.Vari))
}

func (algo *EPLogisticRegression) Init(params map[string]string) {
	algo.Model = make(map[int64]*util.Gaussian)
	algo.params.beta, _ = strconv.ParseFloat(params["beta"], 64)
	algo.params.init_var = 1.0
}

func (algo *EPLogisticRegression) Clear() {
	algo.Model = nil
	algo.Model = make(map[int64]*util.Gaussian)
}

func (algo *EPLogisticRegression) Train(dataset *core.DataSet) {
	// http://tullo.ch/articles/online-learning-with-adpredictor/
	// http://videolectures.net/acml2013_herbrich_real_time_bayesian_learning/
	// http://www.moserware.com/2010/03/computing-your-skill.html
	// https://www.microsoft.com/en-us/research/publication/on-gaussian-expectation-propagation/
	// https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/
	// https://en.wikipedia.org/wiki/Belief_propagation 
	// http://videolectures.net/wapa2010_graepel_tals/
	// Expectation propagation
	// 1. compute initial estimate from all factors
	// repeat 2-4 until all terms converge:
	// 2. q(x) - remove one of the terms from running estimate of posterior (t_i(x))
	// 3. q_new(x) - treat the term removed in 2 as "new" and run ADF update on q(x) and t_i(x)
	//	  Z_new(x) - new normalizing constant			
	// 4. t_i_new(x) - compute this from Z_new(x) and q_new(x) and q(x)
	// 5. Use last Z(x) as estimate of P(D)

	// initial estimate
	for _, sample := range dataset.Samples {
		s := util.Gaussian{Mean: 0.0, Vari: 0.0}
		for _, feature := range sample.Features {
			if feature.Value == 0.0 {
				continue
			}
			wi, ok := algo.Model[feature.Id]
			if !ok {
				wi = &(util.Gaussian{Mean: 0.0, Vari: algo.params.init_var})
				algo.Model[feature.Id] = wi
			}
			// http://mathworld.wolfram.com/NormalSumDistribution.html 
			// see weighted sum
			s.Mean += feature.Value * wi.Mean
			s.Vari += feature.Value * feature.Value * wi.Vari
		}

		// weights -> s -> t
		// 
		// s0 at start is the distribution over existing parameters

		// t2 is the gaussian inside v and w?
		// t2 is a truncated gaussian (who's mean and variance are from s0 + variance from "clutter") 
		//    this is v and w
		// t is s0, but with variance of "clutter" (see EP thesis from Minka) added, and multiplied by 
		//    t2. The multiplication with t2 is the message being sent back to t in the factor graph.
		// s2 is t but with addition of variance of "clutter". It's message from t -> s
		// s is s0 multiplied with s2. Incorporate new "feedback" message

		// gaussian can also be represented in terms of "precision" (1/var) and precision adjusted mean (mean*precision)
		t := s
		t.Vari += algo.params.beta

		t2 := util.Gaussian{Mean: 0.0, Vari: 0.0}
		// https://en.wikipedia.org/wiki/Truncated_normal_distribution
		// this is v and w
		if sample.Label > 0.0 {
			t2.UpperTruncateGaussian(t.Mean, t.Vari, 0.0)
		} else {
			t2.LowerTruncateGaussian(t.Mean, t.Vari, 0.0)
		}
		// combine "result"/t2 (gaussian * step function) with t prediction
		// is this just prior update?
		t.MultGaussian(&t2)
		s2 := t
		// move up from t -> s in factor graph. Add the "click noise"
		s2.Vari += algo.params.beta
		s0 := s
		// combine with old s
		s.MultGaussian(&s2)

		// graph propagation across factors (remove the current feature being updated)
		// see expectation propagation
		for _, feature := range sample.Features {
			if feature.Value == 0.0 {
				continue
			}
			wi0 := util.Gaussian{Mean: 0.0, Vari: algo.params.init_var}
			w2 := util.Gaussian{Mean: 0.0, Vari: 0.0}
			wi, _ := algo.Model[feature.Id]
			// remove the current term/feature's values from s0 (step 2 of EP)
			// message from s-> w_i/feature weight, accounting for forward pass of w->s, but without current feature
			// we divide by the feature.Value (squared for variance) bec. we only want to update the weight, and
			//    in the initial collecting of message data we multiplied by this feature's value.
			w2.Mean = (s.Mean - (s0.Mean - wi.Mean*feature.Value)) / feature.Value
			w2.Vari = (s.Vari + (s0.Vari - wi.Vari*feature.Value*feature.Value)) / (feature.Value * feature.Value)
			wi.MultGaussian(&w2)
			wi_vari := wi.Vari

			// these are eq 14 and 15
			// purpose is to decay past data over time and move it back to prior if no new data
			wi_new_vari := wi_vari * wi0.Vari / (0.99*wi0.Vari + 0.01*wi.Vari)
			wi.Vari = wi_new_vari
			wi.Mean = wi.Vari * (0.99*wi.Mean/wi_vari + 0.01*wi0.Mean/wi.Vari)
			if wi.Vari < algo.params.init_var*0.01 {
				wi.Vari = algo.params.init_var * 0.01
			}
			algo.Model[feature.Id] = wi
		}
	}
}
