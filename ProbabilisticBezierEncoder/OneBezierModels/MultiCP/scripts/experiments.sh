#!/bin/bash


case $1 in
	0)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="5"
		num_transformer_layers="6"
		cp_variance="25"
		variance_drop="0.5"
		epochs_drop="5"
		min_variance="0.8"
		penalization_coef="0.1"
		new_model="True"
		;;
	1)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="4"
		num_transformer_layers="6"
		cp_variance="25"
		variance_drop="0.5"
		epochs_drop="5"
		min_variance="0.8"
		penalization_coef="0.1"
		new_model="True"
		;;
	2)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="5"
		num_transformer_layers="8"
		cp_variance="25"
		variance_drop="0.5"
		epochs_drop="5"
		min_variance="0.8"
		penalization_coef="0.1"
		new_model="True"
		;;
	3)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="5"
		num_transformer_layers="8"
		cp_variance="25"
		variance_drop="0.5"
		epochs_drop="5"
		min_variance="0.8"
		penalization_coef="0.1"
		new_model="True"
		;;
	*)
		echo "NOT SET"
esac
