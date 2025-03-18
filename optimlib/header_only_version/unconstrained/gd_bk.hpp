/*################################################################################
  ##
  ##   Copyright (C) 2016-2022 Keith O'Hara
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

    if (settings.gd_settings.method == 5 || settings.gd_settings.method == 6 || settings.gd_settings.method == 7) {
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

	

    while (grad_err > grad_err_tol && rel_sol_change > rel_sol_change_tol && iter < iter_max &&
		std::abs((obj_val-obj_val_p)/ obj_val)>settings_inp->rel_objfn_change_tol && relChangeResisMax>settings_inp->rel_resis_change_tol) {// add
		settings_inp->iteration = iter;
		++iter;
		
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

        ColVec_t d_p = gd_update(x, grad, grad_p, d, box_objfn, opt_data, iter,
                              gd_settings, adam_vec_m, adam_vec_v);

        ColVec_t x_p = x - d_p;
        grad = grad_p;

		obj_val_p = obj_val;//add 
		
		fp_t par_step_size_p;
		bool isUsePreStepSize = false;

		for (int i = 0; i < settings_inp->resisVec.size(); i++) {
			settings_inp->resisVec_p(i) = settings_inp->resisVec(i); //add
		}

		obj_val = box_objfn(x_p, &grad_p, opt_data);

		fp_t obj_val_preItr = obj_val;//add 
		 
		if (iter==1 && gd_settings.decreaseFactor > 0.0 && gd_settings.decreaseFactor < 1.0) { //Add
			int iterSearchStepSize = 0;
			//while ( (obj_val > obj_val_p&&iterSearchStepSize==0) || (obj_val<obj_val_preItr &&iterSearchStepSize<5)) {
			while (obj_val <= obj_val_preItr) {
			//while (obj_val > obj_val_p && obj_val <= obj_val_preItr){
				std::cout << "Object Function Increasing,, Searching Step Size. Iteration:" << iterSearchStepSize << std::endl;

				adam_vec_m = BMO_MATOPS_ZERO_COLVEC(n_vals);
				adam_vec_v = BMO_MATOPS_ZERO_COLVEC(n_vals);
				
				par_step_size_p = gd_settings.par_step_size;
				isUsePreStepSize = false;

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

				if (obj_val >= obj_val_preItr) {
					isUsePreStepSize = true;
				}

				iterSearchStepSize++;
			}
			if (isUsePreStepSize) {
				gd_settings.par_step_size = par_step_size_p;
				settings_inp->gd_settings.par_step_size = gd_settings.par_step_size;
			}
		}
		//else {
		//	gd_settings.par_step_size = gd_settings.par_step_size*gd_settings.loosenFactor;
		//	settings_inp->gd_settings.par_step_size = gd_settings.par_step_size;
		//}

        if (gd_settings.clip_grad) {
            gradient_clipping(grad_p, gd_settings);
        }

        //

        grad_err = BMO_MATOPS_L2NORM(grad_p);
        rel_sol_change = BMO_MATOPS_L1NORM( BMO_MATOPS_ARRAY_DIV_ARRAY((x_p - x), (BMO_MATOPS_ARRAY_ADD_SCALAR(BMO_MATOPS_ABS(x), OPTIM_FPN_SMALL_NUMBER)) ) );

        d = d_p;
        x = x_p;

        //
		
		//add to print infomation in optimization function
		settings_inp->numOfIteration = iter;

        OPTIM_GD_TRACE(iter-1, grad_err, rel_sol_change, x, d, grad_p, adam_vec_m, adam_vec_v)

		settings_inp->gd_settings.adam_vec_m_p = BMO_MATOPS_ZERO_COLVEC(n_vals);
		settings_inp->gd_settings.adam_vec_v_p = BMO_MATOPS_ZERO_COLVEC(n_vals);
		settings_inp->gd_settings.adam_vec_m_p = adam_vec_m ;
		settings_inp->gd_settings.adam_vec_v_p = adam_vec_v;
		settings_inp->gd_settings.isFirstCalcGD = false;

		relChangeResisMax = std::abs((settings_inp->resisVec(0) - settings_inp->resisVec_p(0)) / settings_inp->resisVec(0));
		for (int i = 1; i < settings_inp->resisVec.size(); i++) {
			double relChangeResis= std::abs((settings_inp->resisVec(i) - settings_inp->resisVec_p(i)) / settings_inp->resisVec(i));
			if (relChangeResis > relChangeResisMax) {
				relChangeResisMax = relChangeResis;
			}
		}
		std::cout << "Relative Max Resistivity Change From Previous Iteration:" << relChangeResisMax << std::endl;

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
