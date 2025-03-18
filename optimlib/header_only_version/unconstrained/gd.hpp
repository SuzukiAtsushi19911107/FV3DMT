/*################################################################################
  ##
  ##   Copyright (C) 2016-2022 Keith O'Hara, revised by Suzuki Atsushi
  ##
  ##   This file is part of the OptimLib C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

/*
 * Gradient Descent (GD)
 */

#ifndef _optim_gd_HPP
#define _optim_gd_HPP
#pragma once
#include <random>
#include <algorithm>
/**
 * @brief The Gradient Descent Optimization Algorithm
 *
 * @param init_out_vals a column vector of initial values, which will be replaced by the solution upon successful completion of the optimization algorithm.
 * @param opt_objfn the function to be minimized, taking three arguments:
 *   - \c vals_inp a vector of inputs;
 *   - \c grad_out a vector to store the gradient; and
 *   - \c opt_data additional data passed to the user-provided function.
 * @param opt_data additional data passed to the user-provided function.
 *
 * @return a boolean value indicating successful completion of the optimization algorithm.
 */

bool 
gd(ColVec_t& init_out_vals, 
   std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
   void* opt_data);

/**
 * @brief The Gradient Descent Optimization Algorithm
 *
 * @param init_out_vals a column vector of initial values, which will be replaced by the solution upon successful completion of the optimization algorithm.
 * @param opt_objfn the function to be minimized, taking three arguments:
 *   - \c vals_inp a vector of inputs;
 *   - \c grad_out a vector to store the gradient; and
 *   - \c opt_data additional data passed to the user-provided function.
 * @param opt_data additional data passed to the user-provided function.
 * @param settings parameters controlling the optimization routine.
 *
 * @return a boolean value indicating successful completion of the optimization algorithm.
 */

bool
gd(ColVec_t& init_out_vals, 
   std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
   void* opt_data, 
   algo_settings_t& settings);

//
// internal

namespace internal
{

bool 
gd_basic_impl(ColVec_t& init_out_vals, 
              std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
              void* opt_data, 
              algo_settings_t* settings_inp);

}

#include "gd.ipp"

//

inline
bool
internal::gd_basic_impl(
    ColVec_t& init_out_vals, 
    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
    void* opt_data, 
    algo_settings_t* settings_inp)
{
    // notation: 'p' stands for '+1'.

    bool success = false;
    
    const size_t n_vals = BMO_MATOPS_SIZE(init_out_vals);

    //
    // GD settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const int print_level = settings.print_level;
    
    const uint_t conv_failure_switch = settings.conv_failure_switch;
    const size_t iter_max = settings.iter_max;
    settings.gd_settings.iter_max = settings.iter_max;
    const fp_t grad_err_tol = settings.grad_err_tol;
    const fp_t rel_sol_change_tol = settings.rel_sol_change_tol;

    gd_settings_t gd_settings = settings.gd_settings;

    const bool vals_bound = settings.vals_bound;
    
    const ColVec_t lower_bounds = settings.lower_bounds;
    const ColVec_t upper_bounds = settings.upper_bounds;

    const ColVecInt_t bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    // lambda function for box constraints

    std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* box_data)> box_objfn \
    = [opt_objfn, vals_bound, bounds_type, lower_bounds, upper_bounds] (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data) \
    -> fp_t 
    {
        if (vals_bound) {
            ColVec_t vals_inv_trans = inv_transform(vals_inp, bounds_type, lower_bounds, upper_bounds);
            
            fp_t ret;
            
            if (grad_out) {
                ColVec_t grad_obj = *grad_out;

                ret = opt_objfn(vals_inv_trans, &grad_obj, opt_data);

                // Mat_t jacob_matrix = jacobian_adjust(vals_inp,bounds_type,lower_bounds,upper_bounds);
                ColVec_t jacob_vec = BMO_MATOPS_EXTRACT_DIAG( jacobian_adjust(vals_inp,bounds_type,lower_bounds,upper_bounds) );

                // *grad_out = jacob_matrix * grad_obj; //
                *grad_out = BMO_MATOPS_HADAMARD_PROD(jacob_vec, grad_obj);
            } else {
                ret = opt_objfn(vals_inv_trans, nullptr, opt_data);
            }

            return ret;
        } else {
            return opt_objfn(vals_inp,grad_out,opt_data);
        }
    };

    //
    // initialization

    if (! BMO_MATOPS_IS_FINITE(init_out_vals) ) {
        printf("gd error: non-finite initial value(s).\n");
        return false;
    }

    ColVec_t x = init_out_vals;
    ColVec_t d = BMO_MATOPS_ZERO_COLVEC(n_vals);

    ColVec_t adam_vec_m;
    ColVec_t adam_vec_v;

    if (settings.gd_settings.method == 3 || settings.gd_settings.method == 4) {
        adam_vec_v = BMO_MATOPS_ZERO_COLVEC(n_vals);
    }

    if (settings.gd_settings.method == 5 || settings.gd_settings.method == 6 || settings.gd_settings.method == 7 || settings.gd_settings.method == 8) {
        adam_vec_m = BMO_MATOPS_ZERO_COLVEC(n_vals);
        adam_vec_v = BMO_MATOPS_ZERO_COLVEC(n_vals);
    }

	if (settings_inp->gd_settings.inheritPreviousSettingAdam) { //add
		if (settings_inp->gd_settings.isFirstCalcGD==false) {
			adam_vec_m = settings_inp->gd_settings.adam_vec_m_p;
			adam_vec_v = settings_inp->gd_settings.adam_vec_v_p;
		}
	}

    if (vals_bound) { // should we transform the parameters?
        x = transform(x, bounds_type, lower_bounds, upper_bounds);
    }

    ColVec_t grad(n_vals); // gradient

	fp_t obj_val; //add
	settings_inp->isFinishOptimize = false;// add
	settings_inp->resisVec= BMO_MATOPS_ZERO_COLVEC(n_vals);// add
	settings_inp->resisVec_p = BMO_MATOPS_ZERO_COLVEC(n_vals);// add
	fp_t relChangeResisMax = 1e30;// add

	obj_val = box_objfn(x,&grad,opt_data);

    fp_t grad_err = BMO_MATOPS_L2NORM(grad);

    OPTIM_GD_TRACE(-1, grad_err, 0.0, x, d, grad, adam_vec_m, adam_vec_v);


    if (grad_err <= grad_err_tol) {
        return true;
    }

    //
    // begin loop

    ColVec_t grad_p = grad;
    fp_t rel_sol_change = 1.0;// add

    size_t iter = 0;

	fp_t obj_val_p = 1e30;//add 

    std::vector<int> shuffleInpedanceArray(settings_inp->useImpedanceDataVec->size());
    std::vector<int> shuffleTipperArray(settings_inp->useTipperDataVec->size());
    for (int i = 0; i < settings_inp->useImpedanceDataVec->size(); i++) {
        shuffleInpedanceArray[i] = i;
    }
    for (int i = 0; i < settings_inp->useTipperDataVec->size(); i++) {
        shuffleTipperArray[i] = i;
    }

    int minIter = gd_settings.minIterations;
    int averageIter = gd_settings.averageIterations;

    double objChange = 1e30;

    std::vector<double> objChangeVector(averageIter);
    std::vector<double> relResisChangeVector(averageIter);

    int numOfBreakWhenIncreasing = 0;

    while (iter<minIter || (grad_err > grad_err_tol  && iter < iter_max &&
		(objChange>settings_inp->rel_objfn_change_tol) && relChangeResisMax>settings_inp->rel_resis_change_tol)) {// add
		settings_inp->iteration = iter;
        settings_inp->numOfIteration = iter;
		++iter;
		//add to print infomation in optimization function
		

		if (settings_inp->isFinishOptimize == true) {
			success = true;
			break;
		}
		
		//if (settings_inp->gd_settings.method == 6 && settings_inp->gd_settings.isRestartAdam == true) {
		//	//adam_vec_m = BMO_MATOPS_ZERO_COLVEC(n_vals);
		//	//adam_vec_v = BMO_MATOPS_ZERO_COLVEC(n_vals);
		//	gd_settings.par_step_size = gd_settings.par_step_size/3;
		//	settings_inp->gd_settings.isRestartAdam = false;
		//	std::cout << "GD Step Size has been Decreased:" << gd_settings.par_step_size << std::endl;
		//}
		//else {
		//	if (gd_settings.par_step_size*1.5 < gd_settings.maxStepSize) {
		//		gd_settings.par_step_size = std::min(gd_settings.par_step_size*1.5, gd_settings.maxStepSize);
		//		settings_inp->gd_settings.par_step_size = gd_settings.par_step_size;
		//	}
		//}
        //
        settings_inp->gd_settings.adam_vec_m_p = BMO_MATOPS_ZERO_COLVEC(n_vals);
        settings_inp->gd_settings.adam_vec_v_p = BMO_MATOPS_ZERO_COLVEC(n_vals);
        settings_inp->gd_settings.adam_vec_m_p = adam_vec_m;
        settings_inp->gd_settings.adam_vec_v_p = adam_vec_v;

        ColVec_t grad_p_pre = BMO_MATOPS_ZERO_COLVEC(n_vals);
        ColVec_t grad_pre = BMO_MATOPS_ZERO_COLVEC(n_vals);

        grad_p_pre = grad_p;
        grad_pre = grad;


        ColVec_t d_p = gd_update(x, grad, grad_p, d, box_objfn, opt_data, iter,
                              gd_settings, adam_vec_m, adam_vec_v);

        ColVec_t x_p = x - d_p;
        grad = grad_p;

		obj_val_p = obj_val;//add 
		
		fp_t par_step_size_p;

		for (int i = 0; i < settings_inp->resisVec.size(); i++) {
			settings_inp->resisVec_p(i) = settings_inp->resisVec(i); //add
		}
        fp_t obj_val_preItr = obj_val;//add 

		obj_val = box_objfn(x_p, &grad_p, opt_data);

        if (obj_val > obj_val_p) {
            numOfBreakWhenIncreasing++;
        }
        else {
            numOfBreakWhenIncreasing = 0;
        }
        if (numOfBreakWhenIncreasing >= gd_settings.numOfBreakWhenIncreasing) {
            std::cout << "Warning:::Objective Function Increased " << gd_settings.numOfBreakWhenIncreasing << " times!!! Exit This Calculation!!!!!" << std::endl;
            break;
        }
		 
		if (iter==1 && gd_settings.decreaseFactor > 0.0 && gd_settings.decreaseFactor < 1.0 && obj_val>= obj_val_preItr) { //Add
			int iterSearchStepSize = 0;
            obj_val_preItr = obj_val;               
			//while ( (obj_val > obj_val_p&&iterSearchStepSize==0) || (obj_val<obj_val_preItr &&iterSearchStepSize<5)) {
			while (iterSearchStepSize<10) {
			//while (obj_val > obj_val_p && obj_val <= obj_val_preItr){
				std::cout << "Object Function Increasing,, Searching Step Size. Iteration:" << iterSearchStepSize << std::endl;

                ColVec_t xPre = BMO_MATOPS_ZERO_COLVEC(n_vals);
                ColVec_t adam_vec_m_pre = BMO_MATOPS_ZERO_COLVEC(n_vals);
                ColVec_t adam_vec_v_pre = BMO_MATOPS_ZERO_COLVEC(n_vals);
                xPre = x_p;
                grad_p = grad_p_pre;
                grad = grad_pre;
                adam_vec_m_pre = adam_vec_m;
                adam_vec_v_pre = adam_vec_v;

                if (settings_inp->gd_settings.inheritPreviousSettingAdam) { //add
                    if (settings_inp->gd_settings.isFirstCalcGD == false) {
                        adam_vec_m = settings_inp->gd_settings.adam_vec_m_p;
                        adam_vec_v = settings_inp->gd_settings.adam_vec_v_p;
                    }
                }
                else {
                    adam_vec_m = BMO_MATOPS_ZERO_COLVEC(n_vals);
                    adam_vec_v = BMO_MATOPS_ZERO_COLVEC(n_vals);
                }
				
				
				par_step_size_p = gd_settings.par_step_size;

				//if (iterSearchStepSize > 0 || iter == 0) {
				//	gd_settings.par_step_size *= gd_settings.decreaseFactor;
				//}
				gd_settings.par_step_size *= gd_settings.decreaseFactor;
				settings_inp->gd_settings.par_step_size = gd_settings.par_step_size;

				d_p = gd_update(x, grad, grad_p, d, box_objfn, opt_data, iter,
					gd_settings, adam_vec_m, adam_vec_v);

				x_p = x - d_p;
				grad = grad_p;

				obj_val_preItr = obj_val;

				obj_val = box_objfn(x_p, &grad_p, opt_data);

                if (obj_val > obj_val_preItr) {
                    x_p = xPre;
                    grad_p = grad_p_pre;
                    grad = grad_pre;
                    if (settings_inp->gd_settings.inheritPreviousSettingAdam) { //add
                        if (settings_inp->gd_settings.isFirstCalcGD == false) {
                            adam_vec_m = settings_inp->gd_settings.adam_vec_m_p;
                            adam_vec_v = settings_inp->gd_settings.adam_vec_v_p;
                        }
                    }
                    else {
                        adam_vec_m = BMO_MATOPS_ZERO_COLVEC(n_vals);
                        adam_vec_v = BMO_MATOPS_ZERO_COLVEC(n_vals);
                    }
                    gd_settings.par_step_size = par_step_size_p;
                    settings_inp->gd_settings.par_step_size = gd_settings.par_step_size;
                    break;
                }

				iterSearchStepSize++;
			}

		}


        if (gd_settings.clip_grad) {
            gradient_clipping(grad_p, gd_settings);
        }

        //

        grad_err = BMO_MATOPS_L2NORM(grad_p);
        rel_sol_change = BMO_MATOPS_L1NORM( BMO_MATOPS_ARRAY_DIV_ARRAY((x_p - x), (BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ABS(x), OPTIM_FPN_SMALL_NUMBER)) ) );

        d = d_p;
        x = x_p;

        //
		
		

        OPTIM_GD_TRACE(iter-1, grad_err, rel_sol_change, x, d, grad_p, adam_vec_m, adam_vec_v)

		
		settings_inp->gd_settings.isFirstCalcGD = false;

		/*relChangeResis = std::abs((settings_inp->resisVec(0) - settings_inp->resisVec_p(0)) / settings_inp->resisVec(0));
		for (int i = 1; i < settings_inp->resisVec.size(); i++) {
			double relChangeResis= std::abs((settings_inp->resisVec(i) - settings_inp->resisVec_p(i)) / settings_inp->resisVec(i));
			if (relChangeResis > relChangeResis) {
				relChangeResis = relChangeResis;
			}
		}*/

        relChangeResisMax = std::abs((settings_inp->resisVec(0) - settings_inp->resisVec_p(0)) / settings_inp->resisVec(0));
        for (int i = 1; i < settings_inp->resisVec.size(); i++) {
            double relChangeResis = std::abs((settings_inp->resisVec(i) - settings_inp->resisVec_p(i)) / settings_inp->resisVec(i));
            if (relChangeResis > relChangeResisMax) {
                relChangeResisMax = relChangeResis;
            }
        }
        std::cout << "Relative Max Resistivity Change From Previous Iteration:" << relChangeResisMax << std::endl;
        //objChange = std::abs((obj_val - obj_val_p) / obj_val);
        if (iter - 1 < gd_settings.numTrunc) {
            objChange = 1e30;
            relChangeResisMax = 1e30;
        }
        else if (iter - 1 - gd_settings.numTrunc < averageIter) {
            relResisChangeVector[iter - 1 - gd_settings.numTrunc] = relChangeResisMax;
            objChangeVector[iter - 1 - gd_settings.numTrunc] = std::abs((obj_val - obj_val_p) / obj_val);
            objChange = 1e30;
            relChangeResisMax = 1e30;
        }
        else {
            for (int i = 0; i < averageIter - 1; i++) {
                relResisChangeVector[i] = relResisChangeVector[i + 1];
                objChangeVector[i] = objChangeVector[i + 1];
            }
            relResisChangeVector[averageIter - 1] = relChangeResisMax;
            objChangeVector[averageIter - 1] = std::abs((obj_val - obj_val_p) / obj_val);

            double aveObjChange = 0.0;
            double aveSolChange = 0.0;
            for (int i = 0; i < averageIter; i++) {
                aveObjChange += objChangeVector[i];
                aveSolChange += relResisChangeVector[i];
            }
            objChange = aveObjChange / averageIter;
            relChangeResisMax = aveSolChange / averageIter;
        }

        //shuffle and set minibatches
        if (settings_inp->gd_settings.minibatches != 1) {
            int iterCount = iter - 1;
            int minibatches = settings_inp->gd_settings.minibatches;
            int batchSizeImpedance = shuffleInpedanceArray.size() / minibatches;
            int batchSizeTipper = shuffleTipperArray.size() / minibatches;
            if ((iterCount + 1) % minibatches == 0) {
                batchSizeImpedance += shuffleInpedanceArray.size() % minibatches; //residual are added
                batchSizeTipper += shuffleTipperArray.size() % minibatches;
            }
            if (iterCount % minibatches == 0) { //shuffle
                std::mt19937 get_rand_mt;
                std::shuffle(shuffleInpedanceArray.begin(), shuffleInpedanceArray.end(), get_rand_mt);
                std::shuffle(shuffleTipperArray.begin(), shuffleTipperArray.end(), get_rand_mt);
            }

            settings_inp->useImpedanceDataVec->setZero();
            settings_inp->useTipperDataVec->setZero();
            int start = int(iterCount % minibatches) * (int(shuffleInpedanceArray.size() / minibatches));
            int end = start + batchSizeImpedance;
            for (int i = start; i < end; i++) {
                settings_inp->useImpedanceDataVec->coeffRef(shuffleInpedanceArray[i]) = 1.0;
            }
            start = int(iterCount % minibatches) * (int(shuffleTipperArray.size() / minibatches));
            end = start + batchSizeTipper;
            for (int i = start; i < end; i++) {
                settings_inp->useTipperDataVec->coeffRef(shuffleTipperArray[i]) = 1.0;
            }
        }
        else {
            settings_inp->useImpedanceDataVec->setOnes();
            settings_inp->useTipperDataVec->setOnes();
        }

        //add setting result to pre
        //*(settings_inp->resultVector_pre) = *(settings_inp->resultVector);
        //*(settings_inp->resultAdjointVector_pre) = *(settings_inp->resultAdjointVector);

    }

    //

    if (vals_bound) {
        x = inv_transform(x, bounds_type, lower_bounds, upper_bounds);
    }

    error_reporting(init_out_vals, x, opt_objfn, opt_data, 
                    success, grad_err, grad_err_tol, iter, iter_max, 
                    conv_failure_switch, settings_inp);

    //
	settings_inp->gd_settings.par_step_size = gd_settings.par_step_size;//Add
	settings_inp->numOfIteration = -1;//add

    return success;
}

inline
bool
gd(ColVec_t& init_out_vals, 
          std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
          void* opt_data)
{
    return internal::gd_basic_impl(init_out_vals,opt_objfn,opt_data,nullptr);
}

inline
bool
gd(ColVec_t& init_out_vals, 
          std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, 
          void* opt_data, 
          algo_settings_t& settings)
{
    return internal::gd_basic_impl(init_out_vals,opt_objfn,opt_data,&settings);
}

#endif
