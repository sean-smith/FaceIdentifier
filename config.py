import identify




d = {
	"train_numbers.xml": {
		'hidden_dim': 32,
		'nb_classes': 10,
		'in_dim': 64,
		'train_func': "train_digits",
		'identify_func': "identify_digits",
	},
	"net_test.xml": {
		'hidden_dim': 150,
		'nb_classes': 40,
		'in_dim': 300,
		'train_func': "train_images",
		'identify_func': "identify3",
		'split_percent': .80,
		'faces_per_person': 10,
	},
	"net.xml": {
		'hidden_dim': 106,
		'nb_classes': 40,
		'in_dim': 10304,
		'train_func': "train_images",
		'identify_func': "identify1",
	},
	"net_sklearn.xml": {
		'hidden_dim': 64,
		'nb_classes': 40,
		'in_dim': 4096,
		'train_func': "train_images2",
		'identify_func': "identify2",
	},
}