#!/bin/bash


case $1 in
	0)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"

		num_control_points="3"
		num_transformer_layers="6"
		max_beziers="2"

		loss_type="probabilisticChamfer"
		dataset_name="MNIST"

		new_model="True"
		;;
	1)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"

		num_control_points="3"
		num_transformer_layers="6"
		max_beziers="2"

		loss_type="authenticChamfer"
		dataset_name="quickdraw"

		new_model="True"
		;;
	2)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"

		num_control_points="3"
		num_transformer_layers="5"
		max_beziers="2"

		loss_type="probabilistic"
		dataset_name="MNIST"

		new_model="True"
		;;
	3)
		batch_size="64"
		num_epochs="500"
		learning_rate="0.00005"

		num_control_points="3"
		num_transformer_layers="5"
		max_beziers="2"

		loss_type="probabilistic"
		dataset_name="MNIST"

		new_model="True"
		;;
	*)
		echo "NOT SET"
esac
