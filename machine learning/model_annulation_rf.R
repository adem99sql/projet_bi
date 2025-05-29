install.packages("randomForest")

# Charger les librairies nécessaires
library(readr)
library(dplyr)
library(janitor)
library(caTools)
library(randomForest)
library(caret)
library(ggplot2)

# Lire les données
df <- read_csv("C:/Users/Mr.KHADHRAOUI/Documents/hotel_analysis/hotel_bookings (1).csv")

# Nettoyage des données
df <- clean_names(df)
df <- na.omit(df)
df <- df[!(df$adults == 0 & df$children == 0 & df$babies == 0), ]
cat("✅ Données après nettoyage :\n")
print(head(df))


# Conversion en facteurs
df$is_canceled <- as.factor(df$is_canceled)
df$hotel <- as.factor(df$hotel)
df$meal <- as.factor(df$meal)
cat("\n✅ Types de variables (facteurs) :\n")
str(df[, c("is_canceled", "hotel", "meal")])

# Normalisation des colonnes numériques
num_cols <- c("lead_time", "adr", "total_of_special_requests")
df[num_cols] <- scale(df[num_cols])
cat("\n✅ Résumé des colonnes normalisées :\n")
summary(df[num_cols])

# Division en train et test
set.seed(123)
split <- sample.split(df$is_canceled, SplitRatio = 0.7)
train <- subset(df, split == TRUE)
test <- subset(df, split == FALSE)
cat("\n✅ Taille des datasets :\n")
cat("Train :", nrow(train), "lignes\n")
cat("Test  :", nrow(test), "lignes\n")

# Entraînement du modèle Random Forest
rf_model <- randomForest(is_canceled ~ lead_time + adr + total_of_special_requests + hotel + meal,
                         data = train, ntree = 100)
# Afficher le résumé du modèle
cat("\n✅ Résumé du modèle Random Forest :\n")
print(rf_model)

# Prédiction
rf_pred <- predict(rf_model, newdata = test)
# Afficher les 10 premières prédictions
cat("\nExemples de prédictions :\n")
print(head(rf_pred, 10))

# Matrice de confusion
cm <- confusionMatrix(rf_pred, test$is_canceled)
print(cm)
# Affichage de l'Accuracy
cat("\n🎯 Accuracy du modèle :", cm$overall['Accuracy'], "\n")

# Importance des variables
varImpPlot(rf_model)

# Préparer les données pour visualiser les prédictions
results_rf <- data.frame(lead_time = test$lead_time,
                         Predicted = rf_pred,
                         Actual = test$is_canceled)

# Visualisation des prédictions sous forme de points
ggplot(results_rf, aes(x = lead_time, color = Predicted)) +
  geom_point(aes(y = as.numeric(Predicted)), alpha = 0.5) +
  labs(title = "Prédictions Random Forest selon Lead Time",
       x = "Lead Time", y = "Annulation Prédit (0=Non, 1=Oui)") +
  theme_minimal()
