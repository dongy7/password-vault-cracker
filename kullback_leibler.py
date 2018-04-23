#--------------------------------------
#     KULLBACK-LEIBLER DIVERGENCE
#--------------------------------------

import sys, math

MINIMAL_PROB = 0.00000000000001

# (max div is 46.5069933284)


#---
# Get the probability of a specific password in a distribution
# If the password is not included, return a very small value
#---
def get_prob(distribution, password):
    if password not in distribution:
        return MINIMAL_PROB
    else:
        return distribution[password]['prob']


#---
# Calculate the Kullback-Leibler divergence (sum).
# p0 and p1: distributions
#---
def calculate_divergence(p0, p1):
    sum = 0.0

    for password in p0:

        # dont include the entry of total passwords
        if (password != '___+ToTaL+___' and password != '___real___'):
            divident = p0[password]['prob']
            divisor = get_prob(p1, password)
            factor = divident

            # error-check: a password not included in one distribution must have the smallest probability
            if (password not in p0
                    and divident > divisor) or (password not in p1
                                                and divident < divisor):
                print("Error: minimal probability too high")
                exit(1)

            # error-check: divisor or divident cannot be 0
            if divisor == 0:
                print(" Error: Divisor is 0")
                exit(1)
            if divident == 0:
                print(" Error: Divident is 0")
                exit(1)

            # only consider divident if divisor is not mininmal
            if divisor == MINIMAL_PROB:
                divident = 1
                # factor = 1

            division = divident / divisor

            # consider number of occurences
            for i in range(0, int(p0[password]["occ"])):
                if 'score' in p0[password]:
                    sum = sum + factor * math.log(division, 2) * p0[password]['score']
                else:
                    sum = sum + factor * math.log(division, 2)

            ####DEBUG#####
            #print "password: "+str(password)
            #print "divident: "+str(divident)
            #print "divisor: "+str(divisor)

    return sum / p0['___+ToTaL+___']["sum"]
