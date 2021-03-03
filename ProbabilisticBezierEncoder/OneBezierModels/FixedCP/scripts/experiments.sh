#!/bin/bash


case $1 in
	0)
		batch_size="64"
		num_epochs="240"
		learning_rate="0.00005"

		num_control_points="5"
		num_transformer_layers="5"

		loss_type="chamfer"
		distance_type="quadratic"

		cp_variance="25"
		variance_drop="0.5"
		epochs_drop="5"
		min_variance="1"
		penalization_coef="0.1"
		new_model="True"
		;;
	1)
		batch_size="64"
		num_epochs="240"
		learning_rate="0.00005"

		num_control_points="5"
		num_transformer_layers="5"

		loss_type="pmap"
		distance_type="quadratic"

		cp_variance="25"
		variance_drop="0.5"
		epochs_drop="5"
		min_variance="1"
		penalization_coef="0.2"
		new_model="True"
		;;
	2)
		batch_size="64"
		num_epochs="240"
		learning_rate="0.00005"

		num_control_points="5"
		num_transformer_layers="5"

		loss_type="pmap"
		distance_type="l2"

		cp_variance="25"
		variance_drop="0.5"
		epochs_drop="5"
		min_variance="1"
		penalization_coef="0.1"
		new_model="True"
		;;
	3)
		batch_size="64"
		num_epochs="240"
		learning_rate="0.00005"

		num_control_points="5"
		num_transformer_layers="5"

		loss_type="pmap"
		distance_type="l2"

		cp_variance="25"
		variance_drop="0.5"
		epochs_drop="5"
		min_variance="1"
		penalization_coef="0.1"
		new_model="True"
		;;
	*)
		echo "NOT SET"
esac
