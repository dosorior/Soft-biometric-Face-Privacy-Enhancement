# Soft-biometric-Face-Privacy-Enhancement

In the recent past, different researchers have proposed privacy-enhancing face recognition systems designed to conceal soft-biometric attributes at feature level. These works have reported impressive results, but generally did not consider specific attacks in their analysis of privacy protection. In this work, we introduce an attack on said schemes based on two observations: (1) highly similar facial representations usually originate from face images with similar soft-biometric attributes; (2) to achieve high recognition accuracy, robustness against intra-class variations within facial representations has to be retained in their privacy-enhanced versions. The presented attack only requires the privacy-enhancing algorithm as a black-box and a relatively small database of face images with annotated soft-biometric attributes. Firstly, an intercepted privacy-enhanced face representation is compared against the attacker's database. Subsequently, the unknown attribute is inferred from the attributes associated with the highest obtained similarity scores. In the experiments, the attack is applied against two state-of-the-art approaches. The attack is shown to circumvent the privacy enhancement to a considerable degree and is able to correctly classify gender with an accuracy of up to approximately 90%. Future works on privacy-enhancing face recognition are encouraged to include the proposed attack in evaluations on the privacy protection.

# Citation

If you use this code in your research, please cite the following paper:

```{bibtex}

@article{OsorioRoig-FaceSoftBiometricPrivacyAttack-TBIOM-2022,
 Author = {D. Osorio-Roig and C. Rathgeb and P Drozdowski and P. Terh{\"o}rst and V. {\v{S}}truc and C. Busch},
 File = {:https\://cased-dms.fbi.h-da.de/literature/OsorioRoig-FaceSoftBiometricPrivacyAttack-TBIOM-2022.pdf:URL},
 Groups = {TReSPAsS-ETN, ATHENE, NGBS},
 Journal = {Trans. on Biometrics, Behavior, and Identity Science ({TBIOM})},
 Keywords = {Soft Biometrics, Face Recognition, Data Privacy},
 Month = {April},
 Number = {2},
 Pages = {263--275},
 Title = {An Attack on Facial Soft-biometric Privacy Enhancement},
 Volume = {4},
 Year = {2022}
}
```

# Contributions

An attack on Soft-biometric privacy-enhanced face templates.

# Attack

![Conceptual Overview of Proposed Attack](images/attack.png)

# Installation

1- Download the databases corresponding to the paper or those databases utilised on the attack. 

2- Prepare the face privacy-enhanced templates to be attacked --> templates should protect the gender as soft-biometric attribute.

3- PE-MIU algorithm (available in https://github.com/pterhoer/PrivacyPreservingFaceRecognition/tree/master/training_free/pe_miu) can be used to generate face privacy-enhanced templates with gender protection.

4- pip install numpy

5- pip install scipy

6- Run script Attacks.py

# Description of parameters

- '-tm', '--testmale', help='path to the male face privacy-enhanced templates'
- '-tf', '--testfemale', help='path to the female face privacy-enhanced templates'
- '-e', '--enrol', help= 'path corresponding to the templates to be enrolled or attacked'
- '-n', '--name', help='soft-biometric to attack, i.e. female or male'
- '-dbf', '--dbfirst', help= 'name of the database to evaluate at search (e.g. the attacker) for cross-database evaluation'
- '-dbs', '--dbsecond', help='name of the database to evaluate at enrolment (e.g. the attacked) for cross-database evaluation'
- '-o', '--output', help='path to the output, file csv with the statistics in terms of similarity scores'


# Suggestions

- Balance the databases in terms of the soft-biometric attribute to be attacked.
- Make sure you have the correct labels corresponding to the soft-biometric attribute from the attacker and attacked database, respectively.
- Prepare the databases according to the known labels.
- You receive with this code the best similarity scores resulting of the comparison between attacker database against attacked database. Similarity scores are optimised according to the used attack. Then, a chance of attack should be computed in terms of percentage for each known label or as average given the soft-biometric attribute to be attacked.



