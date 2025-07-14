

class _CustomEarlyStoppingAndSave(Callback):
	def __init__(self, patience=4, divergence_threshold=0.15):
		super(CustomEarlyStoppingAndSave, self).__init__()
		self.patience = patience
		self.divergence_threshold = divergence_threshold
		# To keep track of the best weights
		self.best_weights = None
		# To keep track of the best accuracy
		self.best_accuracy = 0
		# To keep track of the number of epochs without improvement
		self.wait = 0

	def on_epoch_end(self, epoch, logs=None):
		print(f"Epoch {epoch + 1}: Checking Early Stopping")
		
		if epoch < 15:
			print("Epoch < 15")
			return
		print("Epoch > 15")

		# Extract the training and validation accuracies
		train_accuracy = logs.get('accuracy')
		val_accuracy = logs.get('val_accuracy')

		# Check for divergence
		if abs(train_accuracy - val_accuracy) > self.divergence_threshold:
			print(f'Stopping training due to divergence: |train_accuracy - val_accuracy| > {self.divergence_threshold}')
			self.model.stop_training = True
			return

		# Check for validation accuracy improvement
		if val_accuracy > self.best_accuracy:
			self.best_accuracy = val_accuracy
			self.wait = 0
			# Save the best weights
			self.best_weights = self.model.get_weights()
		else:
			self.wait += 1
			if self.wait >= self.patience:
				print(f'Stopping training due to no improvement in validation accuracy for {self.patience} epochs')
				self.model.stop_training = True
				# Restore the best weights
				if self.best_weights is not None:
					self.model.set_weights(self.best_weights)

	def on_train_end(self, logs=None):
		# If training stops for any reason, restore the best weights
		if self.best_weights is not None:
			self.model.set_weights(self.best_weights)

def _kfold_validation(train_images, train_labels, k, class_labels, learning_rate, epochs, num_filters, filter_size, num_layers, pooling_size, activation_function, batch_size, reg, opt, dropout):
	val_accuracies_fold = []
	train_accuracies_fold = []
	val_accuracies_epoch = []
	train_accuracies_epoch = []
	confusion_matrices = []

	max_val_acc = 0
	best_model = None
	best_epoch = None

	fold_size = len(train_images) // k

	for fold in range(k):
		print(f"Fold: {fold}")

		model = _define_model(num_filters, filter_size, num_layers, pooling_size, activation_function, reg, dropout, 64, class_labels)

		# Define indices for the test set for the current fold
		start_index = fold * fold_size
		end_index = start_index + fold_size

		# Split data into train and test sets for the current fold using slicing
		x_test = train_images[start_index:end_index]
		y_test = train_labels[start_index:end_index]

		# Concatenate the remaining data for the training set using slicing
		x_train = np.concatenate((train_images[:start_index], train_images[end_index:]), axis=0)
		y_train = np.concatenate((train_labels[:start_index], train_labels[end_index:]), axis=0)

		y_train_encoded = tf.keras.utils.to_categorical(y_train, len(class_labels))
		y_test_encoded = tf.keras.utils.to_categorical(y_test, len(class_labels))

		# Compile and train your model
		if opt == "Adam":
			optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
		elif opt == "SGD":
			optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
		elif opt == "Momentum":
			optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
		elif opt == "RMSProp":
			optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

		model.compile(optimizer=optimizer,
					  loss='categorical_crossentropy',
					  metrics=['accuracy'])

		custom_early_stopping_save = _CustomEarlyStoppingAndSave(patience=4, divergence_threshold=0.15)
		history = model.fit(x_train, y_train_encoded, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test_encoded), verbose=0, callbacks = [custom_early_stopping_save])

		# Store the accuracies per fold for training and validation in a list of integers
		val_loss_fold, val_accuracy_fold = model.evaluate(x_test, y_test_encoded, verbose=0)
		val_accuracies_fold.append(val_accuracy_fold)
		train_loss_fold, train_accuracy_fold = model.evaluate(x_train, y_train_encoded, verbose=0)
		train_accuracies_fold.append(train_accuracy_fold)

		# Want to return best fold - keep track of it.
		if val_accuracy_fold > max_val_acc:
			print(f"New Best Acc: {val_accuracy_fold}")
			max_val_acc = val_accuracy_fold
			best_model = model
			best_epoch = len(history.history['accuracy'])
			print(f"Epoch: {best_epoch}")

		# Store the accuracies for each epoch of each fold for training and validation in a list of 1D arrays
		train_accuracies_epoch.append(history.history['accuracy'])
		val_accuracies_epoch.append(history.history['val_accuracy'])

		# Calculate predictions on the validation set
		y_pred = model.predict(x_test, verbose=0)
		y_pred_labels = np.argmax(y_pred, axis=1)
		y_true_labels = np.argmax(y_test_encoded, axis=1)

		# Generate the confusion matrix for the current fold
		confusion_matrices.append({'predicted_values': y_pred_labels, 'ground_truth_values': y_true_labels})

	return val_accuracies_fold, train_accuracies_fold, val_accuracies_epoch, train_accuracies_epoch, confusion_matrices, best_model, best_epoch

def _define_model(num_filters, filter_size, num_layers, pooling_size, activation_function, reg, dropout, img_size, class_labels):
	# Regularization
	if reg == "L1":
		regularization = tf.keras.regularizers.l1(0.01)
	elif reg == "L2":
		regularization = tf.keras.regularizers.l2(0.01)
	else:
		regularization = None

	model = tf.keras.Sequential()
	model.add(layers.Conv2D(num_filters, filter_size, activation=activation_function, input_shape=(img_size, img_size, 3), kernel_regularizer=regularization))
	model.add(layers.MaxPooling2D(pool_size=(pooling_size,pooling_size)))
	for layer in range(num_layers-1):
		model.add(layers.Conv2D(num_filters, filter_size, activation=activation_function, kernel_regularizer=regularization))
		model.add(layers.MaxPooling2D(pool_size=(pooling_size,pooling_size)))
	model.add(layers.Flatten())

	# Dropout
	if dropout > 0:
		model.add(layers.Dropout(dropout))

	model.add(layers.Dense(len(class_labels), activation='softmax', kernel_regularizer=regularization))

	return model

def _plot_confusion_matrix(class_labels: list, confusion_matrices: list, model_name: str) -> np.ndarray:
	# Calculate the sum of confusion matrices
	num_labels = len(class_labels)
	confusion_matrix_sum = np.zeros((num_labels, num_labels), dtype=np.int32)

	for obj in confusion_matrices:
		predicted_values = obj['predicted_values']
		ground_truth_values = obj['ground_truth_values']
		confusion_matrix_sum += confusion_matrix(y_true=ground_truth_values, y_pred=predicted_values)

	# Plot the confusion matrix
	cm = confusion_matrix_sum

	fig, ax = plt.subplots(figsize=(8, 6))
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
	ax.figure.colorbar(im, ax=ax)
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   xticklabels=class_labels, yticklabels=class_labels,
		   ylabel='Ground truth label',
		   xlabel='Predicted label')

	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	filename = model_name + "confusion_matrix.png"
	plt.savefig(filename)
	plt.show()

	return confusion_matrix_sum

def _plot_kfold_val_acc(model_name, training_cycles, val_final_acc):
	# Scatter plot of val_final_acc values
	plt.figure()
	plt.scatter(range(1, training_cycles+1), val_final_acc)
	plt.xlabel('Train Cycle')
	plt.ylabel('Validation Accuracy')
	plt.title(f'Val acc per fold: Rank {model_name} model')
	plt.ylim(0, 1)  # Set y-axis limits to 0 and 1
	avg_val_acc = np.mean(val_final_acc)
	plt.axhline(avg_val_acc, color='r', linestyle='--', label=f'Average Accuracy: {avg_val_acc:.2f}')
	plt.legend()
	filename = model_name + "scatter_plot.png"
	plt.savefig(filename)

def _plot_epoch_train_val_acc(model_name: str, train_accuracies: list, val_accuracies: list, k: int) -> None:
	filename = model_name + "accuracies.png"

	plt.figure(figsize=(10, 6))

	max_epoch = max(max(len(train_acc) for train_acc in train_accuracies),
					max(len(val_acc) for val_acc in val_accuracies))

	# Adjust plotting for variable length epochs
	for i in range(k):
		epochs_train = len(train_accuracies[i])
		epochs_val = len(val_accuracies[i])
		plt.plot(range(1, epochs_train + 1), train_accuracies[i], color="red", alpha=0.2)
		plt.plot(range(1, epochs_val + 1), val_accuracies[i], color="green", alpha=0.2)

	# Calculate mean accuracies per epoch, considering different lengths
	mean_train_acc = [np.mean([train_accuracies[j][i] for j in range(k) if i < len(train_accuracies[j])]) for i in range(max_epoch)]
	mean_val_acc = [np.mean([val_accuracies[j][i] for j in range(k) if i < len(val_accuracies[j])]) for i in range(max_epoch)]

	plt.plot(range(1, max_epoch + 1), mean_train_acc, label='Train Accuracy', color="red", linewidth=2)
	plt.plot(range(1, max_epoch + 1), mean_val_acc, label='Validation Accuracy', color="green", linewidth=2)

	# Labels and title
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title(f'Train/Val Acc Per Epoch: {model_name} Model')
	plt.ylim(0, 1)
	plt.legend()
	plt.savefig(filename)
	plt.show()

def train_model(slurm_array, index, selected_params, train_images, train_labels):
	"""
	Main function
	"""
	print(f"Param set: {selected_params}")
	num_filters = selected_params['num_filters']
	filter_size = selected_params['filter_size']
	learning_rate = selected_params['learning_rate']
	epochs = selected_params['epochs']
	k = selected_params['k']
	num_layers = selected_params['num_layers']
	pooling_size = selected_params['pooling_size']
	activation_function = selected_params['activation_function']
	batch_size = selected_params['batch_size']
	reg = selected_params['reg']
	opt = selected_params['opt']
	dropout = selected_params['dropout']
	class_labels = sorted(np.unique(train_labels))

	print(f"\nModel {index}")
	vaf, taf, vae, tae, cm, best_model, epochs_trained = _kfold_validation(train_images, train_labels, k, class_labels, learning_rate, epochs, num_filters, filter_size, num_layers, pooling_size, activation_function, batch_size, reg, opt, dropout)

	avg_vaf = np.mean(vaf)
	best_vaf, best_taf = np.max(vaf), np.min(taf)
	best_divergence = best_taf-best_vaf

	# Save plots and model
	acc_formatted = f"{(avg_vaf * 100):.2f}".replace(".", "_")
	folder = f"./array_task{slurm_array}/model_{index}_{acc_formatted}/"
	os.makedirs(folder, exist_ok=True)

	_plot_epoch_train_val_acc(folder, tae, vae, k)

	_plot_kfold_val_acc(folder, k, vaf)

	confusion_matrix_sum = _plot_confusion_matrix(class_labels, cm, folder)

	model_path = os.path.join(folder, f'model_{slurm_array}_{index}.keras')
	best_model.save(model_path)

	model_id = f"{slurm_array}_{index}"

	# Save results
	data = {
    "num_layers": num_layers,
    "num_filters": num_filters,
    "filter_size": filter_size,
    "learning_rate": learning_rate,
    "epochs_trained": epochs_trained,
    "k": k,
    "pooling_size": pooling_size,
    "activation_function": activation_function,
    "batch_size": batch_size,
    "reg": reg,
    "opt": opt,
    "dropout": dropout,
    "avg_vaf": avg_vaf,
    "best_vaf": best_vaf,
    "best_divergence": best_divergence,
    "model_id": model_id
	}

	df = pd.DataFrame([data])
	data_path = os.path.join(folder, f'results_{slurm_array}_{index}.csv')
	df.to_csv(data_path, index=False)

	return
