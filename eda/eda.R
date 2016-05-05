# Initial Exploratory Data Analysis

setwd('~/Documents/CS/COMS W4771 - ML/project/eda/')

# Training set, S
# Testing set, T
S <- read.csv('../data/data.csv')
T <- read.csv('../data/quiz.csv')

# 'numeric' columns
nc <- c( 'X2', 'X11', 'X27', 'X28', 'X29', 'X30', 'X31', 'X32', 'X33', 'X34', 
        'X35', 'X36', 'X37', 'X38', 'X39', 'X40', 'X41', 'X42', 'X43', 'X44', 
        'X45', 'X46', 'X47', 'X48', 'X49', 'X50', 'X51', 'X52', 'X53', 'X54', 
        'X55', 'X59', 'X60', 'X62', 'X63', 'X64')

# unique 'numeric' values
uv <- sapply(df[ , nc], function(x) unique(x))


