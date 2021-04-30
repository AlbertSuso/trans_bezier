#!/bin/bash


case $1 in
	0)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"

		num_control_points="3"
		num_transformer_layers="6"
		max_beziers="2"

		rep_coef="0.2"
    dist_thresh="4.5"
    second_term="True"

		new_model="True"
		;;
	1)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"

		num_control_points="3"
		num_transformer_layers="6"
		max_beziers="2"

		rep_coef="0.1"
    dist_thresh="4.5"
    second_term="True"

		new_model="True"
		;;
	2)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"

		num_control_points="3"
		num_transformer_layers="5"
		max_beziers="2"

		rep_coef="0.05"
    dist_thresh="4.5"
    second_term="True"

		new_model="True"
		;;
	3)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"

		num_control_points="3"
		num_transformer_layers="5"
		max_beziers="2"

		rep_coef="0.01"
    dist_thresh="4.5"
    second_term="True"

		new_model="True"
		;;
	*)
		echo "NOT SET"
esac
