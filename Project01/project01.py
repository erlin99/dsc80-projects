
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns 
    a dictionary with the following structure:

    The keys are the general areas of the syllabus: lab, project, 
    midterm, final, disc, checkpoint

    The values are lists that contain the assignment names of that type. 
    For example the lab assignments all have names of the form labXX where XX 
    is a zero-padded two digit number. See the doctests for more details.    

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    
    areas = ['lab', 'project', 'midterm', 'final', 'disc', 'checkpoint']

    # initialize the dictionary 
    names = {}
    for area in areas:
        names[area] = []

    # loop through the columns in grades 
    for col in grades.columns:  

        # check for checkpoint first to avoid confusion with project 
        if 'checkpoint' in col:
            # get indexes where checkpoint appears + padding
            beg = col.index('checkpoint')
            end = beg + len('checkpoint') + 2
            if col[:end] not in names['checkpoint']:
                names['checkpoint'].append(col[:end])

        # check with capital M
        elif 'Midterm' in col:
            if col[:len('Midterm')] not in names['midterm']:
                names['midterm'].append(col)

        #check with capital F
        elif 'Final' in col:
            if col[:len('Final')] not in names['final']:
                names['final'].append(col)
        
        elif 'discussion' in col:
            if col[:len('discussion') + 2] not in names['disc']:
                names['disc'].append(col)
            
        # check for the rest of the areas 
        else:
            for area in ['lab', 'project']:
                if area in col: 
                    if col[:len(area) + 2] not in names[area]:
                        names[area].append(col[:len(area) + 2])
                 
    return names


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def projects_total(grades):
    '''
    projects_total that takes in grades and computes the total project grade
    for the quarter according to the syllabus. 
    The output Series should contain values between 0 and 1.
    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''
    projects = get_assignment_names(grades)['project'] # getting number of projects + names
    
    filtered_cols = [col for col in grades.columns if 'project' in col] # get all columns that include project
    filtered_cols = [col for col in filtered_cols if not('Late' in col or 'check' in col)] # removed irrelevant columns
    filtered_tb = grades[filtered_cols]
    
    cumulative = [] 
    
    for project in projects:
        # getting only columns for current project 
        project_cols = [col for col in filtered_tb.columns if project in col]
        project_tb = filtered_tb[project_cols]
        
        max_cols = [col for col in project_tb.columns if 'Max' in col] # columns for max points 
        max_points = sum(project_tb[max_cols].loc[0]) # max points for current project 
        
        st_grades = project_tb.drop(max_cols, axis=1) # get all student grades for current project 
        total = st_grades.sum(axis=1) # add grades for each student 
        
        percentage = total / max_points # get grade percentage 
        
        if len(cumulative) == 0:
            cumulative = percentage
        else: 
            cumulative += percentage
    
    return cumulative / len(projects)


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

# helper function to calculate time to seconds 
def to_seconds(time):
    times = time.split(':')
    hours = int(times[0])
    minutes = int(times[1])
    seconds = int(times[2])
    
    return hours*60*60 + minutes*60 + seconds

def submitted_by_TA(s):
    pm = 72000
    am = 28800
    day = 86400
    
    if s > 0 and s < 60:
        return True
    
    for i in range(15):
        if am <= s and s <= pm:
            return True
        am += day
        pm += day
    
    return False

def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe 
    grades and a Series indexed by lab assignment that 
    contains the number of submissions that were turned 
    in on time by the student, yet marked 'late' by Gradescope.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """
    
    # get a filtered table with only the labs' lateness columns 
    lab_lateness_cols = [col for col in grades.columns if ('lab' in col and 'Lateness' in col)]
    lab_lateness_tb = grades[lab_lateness_cols]
    
    on_time = {} # dict to keep track of on time submission at each lab 
    
    # go through each column 
    for lab in lab_lateness_tb.columns:
        seconds = lab_lateness_tb[lab].apply(to_seconds) # get the seconds 
        # check if within threshold
        check_on_time = seconds.apply(submitted_by_TA)
        
        on_time[lab[:len('lab') + 2]] = check_on_time.sum()
    
    return pd.Series(on_time)


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------

def lateness_penalty(col):
    """
    lateness_penalty takes in a 'lateness' column and returns 
    a column of penalties according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.8, 0.5}
    True
    """

    one_week = 86400*7 # one week cut
    two_weeks = one_week*2 # second week cut 
    
    seconds = col.apply(to_seconds) # change time to seconds
    
    # helper function that checks penalty for each value 
    def penalty(s):
        # if it wasnt on time 
        if not submitted_by_TA(s):
            if 0 < s and s <= one_week:
                return 0.9
            elif one_week < s and s <= two_weeks:
                return 0.8
            elif s > two_weeks:
                return 0.5
        
        return 1.0
    
    return seconds.apply(penalty)


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def process_labs(grades):
    """
    process_labs that takes in a dataframe like grades and returns
    a dataframe of processed lab scores. The output should:
      * share the same index as grades,
      * have columns given by the lab assignment names (e.g. lab01,...lab10)
      * have values representing the lab grades for each assignment, 
        adjusted for Lateness and scaled to a score between 0 and 1.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1,10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """
    lab_grades = {} 
    lab_names = get_assignment_names(grades)['lab'] # get the lab names 
    
    # loop through the lab names 
    for lab in lab_names:
        # get the columns for the current lab 
        curr_lab = grades[[col for col in grades.columns if (lab in col)]]
        lab_points = curr_lab[lab] # get the lab scores column
        lab_max = curr_lab[lab + ' - Max Points'] # get the max points column
        lab_lateness = curr_lab[lab + ' - Lateness (H:M:S)'] # get the lateness column
        late_penalty = lateness_penalty(lab_lateness) # calculate the late penalty 
        
        # calculate the score and add to the dictionary 
        lab_grades[lab] = ((lab_points / lab_max)*late_penalty).tolist()
    
    return pd.DataFrame(lab_grades) # return df from dictionary 


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def lab_total(processed):
    """
    lab_total takes in dataframe of processed assignments (like the output of 
    Question 5) and computes the total lab grade for each student according to
    the syllabus (returning a Series). 
    
    Your answers should be proportions between 0 and 1.

    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """
    processed = processed.fillna(0) # remove NaN objects
    
    # helper function to calculate total of each student(row)
    def calculate_total(row):
        grades = row.values # get all grades
        grades = np.delete(grades, grades.argmin()) # drop lowest
        return grades.mean() # calculate mean 
    
    return processed.apply(calculate_total, axis=1) 


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def total_points(grades):
    """
    total_points takes in grades and returns the final
    course grades according to the syllabus. Course grades
    should be proportions between zero and one.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    # helper function that calculates the grade of each of the assignments in 
    # names list 
    def process(names):
        all_points = {}
        # loop through name of the assignments 
        for name in names:
            # get columns related to it 
            curr_name = grades[[col for col in grades.columns if (name in col)]]
            points = curr_name[name] # get points column
            max_points = curr_name[name + ' - Max Points'] # get max points col 
            
            all_points[name] = (points / max_points).tolist()
        
        return pd.DataFrame(all_points)

    # helper function to calculate the total of each student (row) 
    def calculate_total(processed):
        def student_total(row):
            grades = row.values # get all grades
            return grades.mean() # calculate mean 

        return processed.apply(student_total, axis=1)

    grades = grades.fillna(0)

    lab_grades = lab_total(process_labs(grades)) 
    project_grades = projects_total(grades)

    # get names of the graded topics 
    names = get_assignment_names(grades)
    discussion_names= names['disc'] # get discussion names 
    checkpoint_names = names['checkpoint'] # get checkpoint names 

    disc_grades = calculate_total(process(discussion_names)) 
    check_grades = calculate_total(process(checkpoint_names))
    final_grades = calculate_total(process(['Final']))
    midterm_grades = calculate_total(process(['Midterm']))

    # calculate total of each student 
    total = (lab_grades*0.2 + project_grades*0.3 + disc_grades*0.025 + 
        check_grades*0.025 + final_grades*0.3 + midterm_grades*0.15)

    return total


def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.

    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """
    def letter_grade(grade):
        if grade >= .9:
            return 'A'
        elif grade >= .8:
            return 'B'
        elif grade >= .7:
            return 'C'
        elif grade >= .6:
            return 'D'
        else:
            return 'F'
        
    return total.apply(letter_grade)


def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades 
    and outputs a Series that contains the proportion
    of the class that received each grade.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """
    total = total_points(grades)
    letter_grades = final_grades(total)

    return letter_grades.value_counts() / letter_grades.count()

# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def simulate_pval(grades, N):
    """
    simulate_pval takes in the number of
    simulations N and grades and returns
    the likelihood that the grade of sophomores
    was no better on average than the class
    as a whole (i.e. calculate the p-value).

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 100)
    >>> 0 <= out <= 0.1
    True
    """

    total = total_points(grades) # get the total grades of each student 

    # create new df with students level 
    df = pd.DataFrame({'Level': grades['Level'], 'Grades': total})
    obs_avg = np.mean(df[df['Level'] == 'SO']['Grades'])

    averages = []
    # number of sophomore students 
    size = grades['Level'].value_counts()['SO']
    # N simulations 
    for _ in range(N):
        random_sample = total.sample(size, replace=False)
        average = np.mean(random_sample)
        averages.append(average)

    averages = np.array(averages)

    # return p-value 
    return np.count_nonzero(averages >= obs_avg) / N


# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------

def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades, 
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    # helper function that adds noise to the grades in the given dataframe 
    def add_noise(grades):
        grades = grades.fillna(0) # replace NaN values 
        names = get_assignment_names(grades) # get assignment names 
        # go through each name 
        for lst in names.values():
            for name in lst:
                # add noise to the values in each column 
                with_noise = grades[name] + (np.random.normal(0, 0.02, size=grades[name].size)*100)
                grades[name] = np.clip(with_noise, 0, 100)
        
        return grades
    
    grades = add_noise(grades) # add noise 
    total = total_points(grades) # calculate total with the added noise 
    return total


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def short_answer():
    """
    short_answer returns (hard-coded) answers to the 
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.

    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 50 < out[2][0] < 100
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4], bool)
    True
    """

    return [0.0010899392772590888, 64.85, [61, 69.54], 
        0.10467289719626169, False]

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_assignment_names'],
    'q02': ['projects_total'],
    'q03': ['last_minute_submissions'],
    'q04': ['lateness_penalty'],
    'q05': ['process_labs'],
    'q06': ['lab_total'],
    'q07': ['total_points', 'final_grades', 'letter_proportions'],
    'q08': ['simulate_pval'],
    'q09': ['total_points_with_noise'],
    'q10': ['short_answer']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
