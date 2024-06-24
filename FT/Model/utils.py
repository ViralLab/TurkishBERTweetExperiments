from sklearn.model_selection import StratifiedKFold, train_test_split


class MyStratifiedKFold(StratifiedKFold):
    def split(self, x, y, dev=False, groups=None):
        s = super().split(x, y, groups)
        if dev == True:
            for train_indices, test_indices in s:
                y_train = y[train_indices]
                train_indices, dev_indices = train_test_split(
                    train_indices, stratify=y_train, test_size=(1 / (self.n_splits - 1))
                )

                yield (train_indices, dev_indices, test_indices)

        else:
            for train_indices, test_indices in s:
                yield (train_indices, test_indices)


from sklearn.model_selection import StratifiedKFold, train_test_split


class MyStratifiedKFold(StratifiedKFold):
    def split(self, x, y, dev=False, groups=None):
        s = super().split(x, y, groups)
        if dev == True:
            for train_indices, test_indices in s:
                y_train = y[train_indices]
                train_indices, dev_indices = train_test_split(
                    train_indices, stratify=y_train, test_size=(1 / (self.n_splits - 1))
                )

                yield (train_indices, dev_indices, test_indices)

        else:
            for train_indices, test_indices in s:
                yield (train_indices, test_indices)
