#!/bin/bash


case $1 in
	0)
		batch_size="64"
		num_epochs="100"
		learning_rate="0.00005"
		num_control_points="3"
		num_transformer_layers="4"
		trans_encoder="True"
		;;
	1)
		batch_size="64"
		num_epochs="150"
		learning_rate="0.00005"
		num_control_points="4"
		num_transformer_layers="6"
		trans_encoder="True"
		;;
	2)
		batch_size="64"
		num_epochs="200"
		learning_rate="0.00005"
		num_control_points="5"
		num_transformer_layers="8"
		trans_encoder="True"
		;;
	3)
		batch_size="64"
		num_epochs="250"
		learning_rate="0.00005"
		num_control_points="6"
		num_transformer_layers="8"
		trans_encoder="True"
		;;
	*)
		echo "NOT SET"
esac
