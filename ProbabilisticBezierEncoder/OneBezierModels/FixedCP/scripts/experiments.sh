#!/bin/bash


case $1 in
	0)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="3"
		num_transformer_layers="6"
		control_points_variance="5"
		new_model="True"
		trans_encoder="True"
		;;
	1)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="3"
		num_transformer_layers="6"
		control_points_variance="20"
		new_model="True"
		trans_encoder="True"
		;;
	2)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="3"
		num_transformer_layers="6"
		control_points_variance="35"
		new_model="True"
		trans_encoder="True"
		;;
	3)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="3"
		num_transformer_layers="6"
		control_points_variance="50"
		new_model="True"
		trans_encoder="True"
		;;
	*)
		echo "NOT SET"
esac
