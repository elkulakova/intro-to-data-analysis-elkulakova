import unittest
import pandas as pd
from cs102_pandas import (
    filter_fsuir_students, find_homonymous_students, analyze_patronyms,
    faculty_statistics, course_statistics, most_popular_name,
    find_students_with_name_starting_P, highest_avg_grade_faculty, find_consecutive_students
)

class TestDataAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv("isu_fake_data.csv")
        cls.data["surname"] = cls.data["фио"].apply(lambda x: x.split()[0])
        cls.data["name"] = cls.data["фио"].apply(lambda x: x.split()[1])
        cls.data["patronim"] = cls.data["фио"].apply(lambda x: x.split()[2] if len(x.split()) > 2 else "")
    
    def test_filter_fsuir_students(self):
        num_students, num_groups, _ = filter_fsuir_students(self.data)
        self.assertEqual(num_students, 993)
        self.assertEqual(num_groups, 56)
    
    def test_find_homonymous_students(self):
        _, _, fsuir = filter_fsuir_students(self.data)
        has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group = find_homonymous_students(fsuir)
        self.assertTrue(has_homonyms)
        self.assertEqual(total_homonyms, 451)
        self.assertEqual(homonyms_per_course.loc["1-й"], 111)
        self.assertEqual(homonyms_per_course.loc["2-й"], 111)
        self.assertEqual(homonyms_per_course.loc["3-й"], 134)
        self.assertEqual(homonyms_per_course.loc["4-й"], 94)
        self.assertEqual(max_homonym_group, "R33441c")
    
    def test_analyze_patronyms(self):
        _, _, fsuir = filter_fsuir_students(self.data)
        students_without_patronym, gender_counts = analyze_patronyms(fsuir)
        self.assertEqual(students_without_patronym, 409)
        self.assertEqual(gender_counts["male"], 384)
        self.assertEqual(gender_counts["female"], 155)
    
    def test_faculty_statistics(self):
        _, max_faculty, min_faculty = faculty_statistics(self.data)
        self.assertEqual(max_faculty[0], "факультет программной инженерии и компьютерной техники")
        self.assertEqual(max_faculty[1], 2154)
        self.assertEqual(min_faculty[0], "институт международного развития и партнерства")
        self.assertEqual(min_faculty[1], 26)
    
    def test_course_statistics(self):
        mean_students, median_students = course_statistics(self.data)
        self.assertEqual(mean_students.loc["1-й"], 215.0)
        self.assertEqual(mean_students.loc["2-й"], 205.1)
        self.assertEqual(mean_students.loc["3-й"], 166.0)
        self.assertEqual(mean_students.loc["4-й"], 139.3)
        self.assertEqual(median_students.loc["1-й"], 79.0)
        self.assertEqual(median_students.loc["2-й"], 118.0)
        self.assertEqual(median_students.loc["3-й"], 98.0)
        self.assertEqual(median_students.loc["4-й"], 95.5)
    
    def test_most_popular_name(self):
        popular_name, name_group, faculty, course, name_ratio = most_popular_name(self.data)
        self.assertEqual(popular_name, "Александр")
        self.assertEqual(name_group, "K3241")
        self.assertEqual(faculty, "факультет инфокоммуникационных технологий")
        self.assertEqual(course, "2-й")
        self.assertEqual(name_ratio, 0.04)

    def test_find_students_with_name_starting_P(self):
        result_7 = find_students_with_name_starting_P(self.data)
        self.assertEqual(result_7.shape, (16, 3))
        self.assertTrue(result_7["фио"].str.startswith("П").any())
        expected_students = [
            ("Арсланова Патина Гасановна", "факультет инфокоммуникационных технологий", "2-й"),
            ("Круглов Принц Эммануэль", "факультет программной инженерии и компьютерной техники", "1-й")
        ]
        for student in expected_students:
            self.assertTrue(any(result_7["фио"] == student[0]))
            self.assertTrue(any(result_7["факультет"] == student[1]))
            self.assertTrue(any(result_7["курс"] == student[2]))

    def test_highest_avg_grade_faculty(self):
        fac, best_gender, best_grade = highest_avg_grade_faculty(self.data)
        self.assertEqual(fac, "физический факультет")
        self.assertEqual(best_gender, "female")
        self.assertEqual(best_grade, 86)

    def test_find_consecutive_students(self):
        result_9 = find_consecutive_students(self.data)
        self.assertEqual(result_9.shape[0], 5)
        self.assertTrue(all(result_9["ису"].diff().dropna() == 1))
        
        expected_isu = [311121, 311122, 311123, 311124, 311125]
        for isu in expected_isu:
            student_data = result_9[result_9["ису"] == isu]
            self.assertEqual(student_data["группа"].iloc[0], "R34351" if isu == 311121 else
                                                     "Z34431" if isu == 311122 else
                                                     "M34011" if isu == 311123 else
                                                     "M34041" if isu == 311124 else "N34461")
            self.assertEqual(student_data["курс"].iloc[0], "4-й")


if __name__ == "__main__":
    unittest.main()
