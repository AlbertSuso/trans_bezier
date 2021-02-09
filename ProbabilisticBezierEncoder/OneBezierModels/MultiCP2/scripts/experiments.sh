#!/bin/bash


case $1 in
	0)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="3"
		cp_variance="25"
		variance_drop="0.5"
		epochs_drop="10"
		min_variance="0.8"
		new_model="True"
		;;
	1)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="5"
		cp_variance="25"
		variance_drop="0.5"
		epochs_drop="10"
		min_variance="0.8"
		new_model="True"
		;;
	2)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="5"
		cp_variance="25"
		variance_drop="0.5"
		epochs_drop="10"
		min_variance="0.8"
		new_model="True"
		;;
	3)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="5"
		cp_variance="25"
		variance_drop="0.5"
		epochs_drop="10"
		min_variance="0.8"
		new_model="True"
		;;
	*)
		echo "NOT SET"
esac
