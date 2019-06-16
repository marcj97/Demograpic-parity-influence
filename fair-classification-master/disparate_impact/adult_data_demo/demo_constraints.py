import os,sys
import numpy as np
from prepare_adult_data import *
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import utils as ut
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints



def test_adult_data():
	

	""" Load the adult data """
	X, y, x_control = load_adult_data(load_data_size=None) # set the argument to none, or no arguments if you want to test with the whole data -- we are subsampling for performance speedup
	ut.compute_p_rule(x_control["sex"], y) # compute the p-rule in the original data

	""" Split the data into train and test """
	X = ut.add_intercept(X) # add intercept to X before applying the linear classifier
	train_fold_size = 0.7
	x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

	apply_fairness_constraints = None
	apply_accuracy_constraint = None
	sep_constraint = None

	loss_function = lf._logistic_loss
	sensitive_attrs = ["sex"]
	sensitive_attrs_to_cov_thresh = {}
	gamma = None

	def train_test_classifier():
		w = ut.train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
		train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w, x_train, y_train, x_test, y_test, None, None)
		distances_boundary_test = (np.dot(x_test, w)).tolist()
		all_class_labels_assigned_test = np.sign(distances_boundary_test)
		correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
		cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs)
		p_rule = ut.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])
		eq_op_acc, chance_bin_zero, chance_bin_one = ut.get_eq_op_acc(w, x_train, y_train, x_control_train, None)
		eq_odds_acc = ut.get_eq_odds_acc(w, x_train, y_train, x_control_train, None)
		pred_rate_par_acc = ut.get_pred_rate_par_acc(w, x_train, y_train, x_control_train, None)
		demo_par_acc_f_cons = ut.get_dem_par_acc(w, x_train, y_train, x_control_train, None)
		return w, p_rule, test_score, eq_op_acc, eq_odds_acc, pred_rate_par_acc,demo_par_acc_f_cons

	""" Classify the data while optimizing for accuracy """
	print()
	print("== Unconstrained (original) classifier ==")
	# all constraint flags are set to 0 since we want to train an unconstrained (original) classifier
	apply_fairness_constraints = 0
	apply_accuracy_constraint = 0
	sep_constraint = 0
	w_uncons, p_uncons, acc_uncons, eq_op_acc_uncons, eq_odds_acc_uncons, pred_rate_par_acc_uncons, demo_par_acc_uncons = train_test_classifier()

	temp_eq_op_acc_f = []
	temp_eq_odds_acc_f = []
	temp_pred_rate_par_acc_f = []
	temp_demo_par_acc_f = []

	""" Now classify such that we optimize for accuracy while achieving perfect fairness """
	apply_fairness_constraints = 1 # set this flag to one since we want to optimize accuracy subject to fairness constraints
	apply_accuracy_constraint = 0
	sep_constraint = 0
	for num in np.arange(0,0.51,0.1):
		sensitive_attrs_to_cov_thresh = {"sex":num}
		print()
		print("== Classifier with fairness constraint, cov: ",num ," ==")
		w_f_cons, p_f_cons, acc_f_cons, eq_op_acc_f_cons, eq_odds_acc_f_cons, pred_rate_par_acc_f_cons, demo_par_acc_f_cons = train_test_classifier()
		temp_eq_op_acc_f.append(eq_op_acc_f_cons)
		temp_eq_odds_acc_f.append(eq_odds_acc_f_cons)
		temp_pred_rate_par_acc_f.append(pred_rate_par_acc_f_cons)
		temp_demo_par_acc_f.append(demo_par_acc_f_cons)

	sensitive_attrs_to_cov_thresh = {"sex":1}
	print()
	print("== Classifier with fairness constraint, cov: 1 ==")
	w_f_cons, p_f_cons, acc_f_cons, eq_op_acc_f_cons, eq_odds_acc_f_cons, pred_rate_par_acc_f_cons, demo_par_acc_f_cons = train_test_classifier()
	temp_eq_op_acc_f.append(eq_op_acc_f_cons)
	temp_eq_odds_acc_f.append(eq_odds_acc_f_cons)
	temp_pred_rate_par_acc_f.append(pred_rate_par_acc_f_cons)
	temp_demo_par_acc_f.append(demo_par_acc_f_cons)
	
	return eq_op_acc_uncons, eq_odds_acc_uncons, pred_rate_par_acc_uncons, demo_par_acc_uncons,temp_eq_op_acc_f,temp_eq_odds_acc_f,temp_pred_rate_par_acc_f,temp_demo_par_acc_f

def main():
	iterations = 20   
	eq_op_acc_un = []
	eq_odds_acc_un = []
	pred_rate_par_acc_un = []
	demo_par_acc_un = []
	
	eq_op_acc_f = {}
	eq_odds_acc_f = {}
	pred_rate_par_acc_f = {}
	demo_par_acc_f = {}
	
	labels = ["0", ".1", ".2", ".3", ".4", ".5", "1"]
	
	for label in labels:
		eq_op_acc_f[label] = []
		eq_odds_acc_f[label] = []
		pred_rate_par_acc_f[label] = []
		demo_par_acc_f[label] = []
	
	for i in range(iterations):
		print(i+1)
		eq_op_acc_uncons, eq_odds_acc_uncons, pred_rate_par_acc_uncons,demo_par_acc_uncons,temp_eq_op_acc_f,temp_eq_odds_acc_f,temp_pred_rate_par_acc_f,temp_demo_par_acc_f = test_adult_data()
		
		eq_op_acc_un.append(eq_op_acc_uncons)
		eq_odds_acc_un.append(eq_odds_acc_uncons)
		pred_rate_par_acc_un.append(pred_rate_par_acc_uncons)
		demo_par_acc_un.append(demo_par_acc_uncons)
		
		for j in range(len(temp_eq_op_acc_f)):
			eq_op_acc_f[labels[j]].append(temp_eq_op_acc_f[j])
			eq_odds_acc_f[labels[j]].append(temp_eq_odds_acc_f[j])
			pred_rate_par_acc_f[labels[j]].append(temp_pred_rate_par_acc_f[j])
			demo_par_acc_f[labels[j]].append(temp_demo_par_acc_f[j])

	print("equality of opportunity:")
	print("\t unconstrained: \t mean:", np.mean(eq_op_acc_un), "\t std: ", np.std(eq_op_acc_un))
	print("\t fairness constrained:")
	for label in labels:
		print("cov: ", label, "\t mean: ", np.mean(eq_op_acc_f[label]), "\t std: ", np.std(eq_op_acc_f[label]))
        
	print()  
      
	print("equalized odds:")
	print("\t unconstrained: \t mean:", np.mean(eq_odds_acc_un), "\t std: ", np.std(eq_odds_acc_un))
	print("\t fairness constrained:")
	for label in labels:
		print("cov: ", label, "\t mean: ", np.mean(eq_odds_acc_f[label]), "\t std: ", np.std(eq_odds_acc_f[label]))

	print() 
        
	print("predictive rate parity:")
	print("\t unconstrained: \t mean:", np.mean(pred_rate_par_acc_un), "\t std: ", np.std(pred_rate_par_acc_un))
	print("\t fairness constrained:")
	for label in labels:
		print("cov: ", label, "\t mean: ", np.mean(pred_rate_par_acc_f[label]), "\t std: ", np.std(pred_rate_par_acc_f[label]))
       
	print() 
        
	print("demographic parity:")
	print("\t unconstrained: \t mean:", np.mean(demo_par_acc_un), "\t std: ", np.std(demo_par_acc_un))
	print("\t fairness constrained:")
	for label in labels:
		print("cov: ", label, "\t mean: ", np.mean(demo_par_acc_f[label]), "\t std: ", np.std(demo_par_acc_f[label]))
        


if __name__ == '__main__':
	main()