#!/bin/bash


case $1 in
	0)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="3"
		max_beziers="2"
		num_transformer_layers="8"
		new_model="True"
		trans_encoder="True"
		;;
	1)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="7"
		max_beziers="2"
		num_transformer_layers="8"
		new_model="True"
		trans_encoder="True"
		;;
	2)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"
		num_control_points="8"
		max_beziers="2"
		num_transformer_layers="8"
		new_model="True"
		trans_encoder="True"
		;;
	3)
		batch_size="64"
		num_epochs="250"
		learning_rate="0.00005"
		num_control_points="6"
		max_beziers="2"
		num_transformer_layers="8"
		new_model="True"
		trans_encoder="True"
		;;
	*)
		echo "NOT SET"
esac
