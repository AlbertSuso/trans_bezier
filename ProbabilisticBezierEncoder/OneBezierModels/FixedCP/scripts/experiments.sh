#!/bin/bash


case $1 in
	0)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="3"
		num_transformer_layers="6"
		new_model="True"
		;;
	1)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="3"
		num_transformer_layers="6"
		new_model="True"
		;;
	2)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="3"
		num_transformer_layers="6"
		new_model="True"
		;;
	3)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="3"
		num_transformer_layers="6"
		new_model="True"
		;;
	*)
		echo "NOT SET"
esac
