
# Charger la librairie readr (pour lire le fichier .csv)
library(readr)

# Lire le fichier CSV (remplace le chemin si nécessaire)
df <- read_csv("C:/Users/Mr.KHADHRAOUI/Documents/hotel_analysis/hotel_bookings (1).csv")

# Afficher les premières lignes
head(df)

# Charger dplyr et janitor pour manipuler et nettoyer
library(dplyr)
library(janitor)

# Nettoyer les noms de colonnes (en minuscules, sans espaces)
df <- clean_names(df)

# Voir les colonnes et types
glimpse(df)

# Voir le total de NA par colonne
colSums(is.na(df))

# Supprimer les lignes avec NA (option simple)
df <- na.omit(df)

# OU : remplacer les NA dans une colonne (exemple : children)
df$children[is.na(df$children)] <- median(df$children, na.rm = TRUE)

# Supprimer les lignes avec 0 adultes, 0 enfants, et 0 bébés
df <- df[!(df$adults == 0 & df$children == 0 & df$babies == 0), ]

# Transformer certaines colonnes en facteurs (catégories)
df$is_canceled <- as.factor(df$is_canceled)
df$hotel <- as.factor(df$hotel)
df$meal <- as.factor(df$meal)

# Choisir les colonnes numériques importantes
num_cols <- c("lead_time", "adr", "total_of_special_requests")

# Appliquer la normalisation
df[num_cols] <- scale(df[num_cols])
summary(df[num_cols])

# Charger le package pour split
library(caTools)

# Fixer la graine pour la reproductibilité
set.seed(123)

# Créer la division
split <- sample.split(df$is_canceled, SplitRatio = 0.7)

# Créer les datasets
train <- subset(df, split == TRUE)
test <- subset(df, split == FALSE)

# KNN - Variables explicatives (features)
library(class)
library(caret)
X_train <- train[, c("lead_time", "adr", "total_of_special_requests")]
X_test <- test[, c("lead_time", "adr", "total_of_special_requests")]
y_train <- train$is_canceled
y_test <- test$is_canceled

# k = 5 par exemple
knn_pred <- knn(train = X_train, test = X_test, cl = y_train, k = 5)
confusionMatrix(knn_pred, y_test)

# Tester différents k
accuracies <- c()
for (k in 1:20) {
  pred_k <- knn(train = X_train, test = X_test, cl = y_train, k = k)
  cm <- confusionMatrix(pred_k, y_test)
  acc <- cm$overall['Accuracy']
  accuracies <- c(accuracies, acc)
}
print(accuracies)
which.max(accuracies)

# Matrice de confusion graphique
cm <- confusionMatrix(knn_pred, y_test)
cm_data <- as.data.frame(as.table(cm))
colnames(cm_data) <- c("Predicted", "Actual", "Frequency")
library(ggplot2)
ggplot(cm_data, aes(x = Actual, y = Predicted, fill = Frequency)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Matrice de Confusion", x = "Valeur Réelle", y = "Prédiction") +
  theme_minimal()

# Régression logistique
lr_model <- glm(is_canceled ~ lead_time + adr + total_of_special_requests,
                family = binomial(link = 'logit'), data = train)
lr_prob <- predict(lr_model, test, type = "response")


# Graphique de probabilité par lead_time
results_lr <- data.frame(lead_time = test$lead_time,
                         Predicted_Prob = lr_prob,
                         Actual = y_test)
ggplot(results_lr, aes(x = lead_time, y = Predicted_Prob, color = factor(Actual))) +
  geom_point(alpha = 0.6) +
  labs(title = "Prédictions des Probabilités par Lead Time (Régression Logistique)",
       x = "Lead Time",
       y = "Probabilité Prédite") +
  scale_color_manual(values = c("blue", "red"), name = "Annulation",
                     labels = c("Non Annulé", "Annulé")) +
  theme_minimal() 
