"""Обработка файла с данными ИСУ"""

import re
from typing import Dict, Tuple

import pandas as pd


# Задача 1
def filter_fsuir_students(data: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    """
    Создает подвыборку студентов факультета систем управления и робототехники (ФСУиР).
    Возвращает количество таких студентов, количество уникальных групп и отфильтрованный датасет.
    """
    fsuirs = data[data["факультет"] == "факультет систем управления и робототехники"]
    students_numb = len(fsuirs)
    groups_numb = len(fsuirs["группа"].unique())
    return students_numb, groups_numb, fsuirs


# Задача 2
def find_homonymous_students(df: pd.DataFrame) -> Tuple[bool, int, pd.Series, str]:
    """
    Проверяет наличие однофамильцев на ФСУиР, их количество, распределение по курсам
    и определяет группу с наибольшим числом однофамильцев.
    Возвращает:
     - логическое значение (наличие однофамильцев)
     - общее количество однофамильцев
     - серию с числом однофамильцев по курсам
     - группу с максимальным числом однофамильцев
    """
    surnames = df.surname.value_counts()
    exist = any(numb > 1 for numb in surnames)
    homonyms_numb = sum(numb for numb in surnames if numb > 1)

    df = df.sort_values(by="курс", ascending=True)
    courses = sorted(df.курс.unique())
    hom_on_course = pd.Series(
        [sum(numb for numb in df[df.курс == c].surname.value_counts() if numb > 1) for c in courses],
        index=courses,
    )

    df = df.sort_values(by="группа", ascending=True)
    groups = df.группа.unique()
    groups_with_homs = {
        group: sum(numb for numb in df[df.группа == group].surname.value_counts() if numb > 1) for group in groups
    }
    max_hom_group = max(groups_with_homs, key=groups_with_homs.get)

    return exist, homonyms_numb, hom_on_course, max_hom_group


# Задача 3
def gender_identification(patronym: str) -> str:
    """
    Определяет пол по отчеству. Возвращает пол: female/male/unknown.
    """
    # В современном русском языке мужские отчества имеют окончания -ович/-евич/-ич, женские – -овна/-евна/-ична/-инична.

    female_patr = r"\b[А-Яа-я]+(евна|овна|ична|инична)\b"
    male_patr = r"^\b[А-Яа-я-–]*[А-Яа-я]+(ович|евич|ич)\b"  # Руслан-Бекович!!

    if re.search(female_patr, patronym):
        return "female"
    if re.search(male_patr, patronym):
        return "male"

    return "unknown"


def analyze_patronyms(df: pd.DataFrame) -> Tuple[int, pd.Series]:
    """
    Определяет количество студентов без отчества и распределение студентов по полу на основе отчества.
    Возвращает:
     - количество студентов без отчества
     - серию с распределением студентов по полу
    """
    stud_without_patr = len([patr for patr in df.patronim if not patr])

    stud_with_patr = df[df.patronim != ""]
    genders = stud_with_patr.patronim.apply(gender_identification).value_counts()

    return stud_without_patr, genders


# Задача 4
def faculty_statistics(data: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[str, int], Tuple[str, int]]:
    """
    Подсчитывает количество студентов на каждом факультете,
    а также определяет факультеты с максимальным и минимальным числом студентов.
    """
    faculties = data.факультет.value_counts()
    stats = pd.DataFrame(
        {"faculty": faculties.index, "students_numb": faculties.values}, index=range(1, len(faculties) + 1)
    )
    min_faclt, min_numb = stats.at[len(faculties), "faculty"], stats.at[len(faculties), "students_numb"]
    max_faclt, max_numb = stats.at[1, "faculty"], stats.at[1, "students_numb"]

    return stats, (max_faclt, max_numb), (min_faclt, min_numb)


# Задача 5
def course_statistics(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Вычисляет среднее и медианное число студентов на каждом курсе.
    Возвращает две серии с результатами: сначала средние, потом медиана.
    """
    courses = sorted(data.курс.unique())
    stats_c_av = pd.Series([data[data.курс == cur].факультет.value_counts().mean() for cur in courses], index=courses)
    stats_c_med = pd.Series(
        [data[data["курс"] == cur]["факультет"].value_counts().median() for cur in courses], index=courses
    )

    return stats_c_av, stats_c_med


# Задача 6
def most_popular_name(data: pd.DataFrame) -> Tuple[str, str, str, int, float]:
    """
    Определяет самое популярное имя, группу с наибольшим количеством студентов с этим именем,
    факультет, курс и долю таких студентов в общем числе.
    Возвращает результат в следующем порядке:
     1. самое частое имя
     2. группа
     3. факультет
     4. курс
     5. доля
    """
    top_name: str = data.name.describe()["top"]
    top_group: str = data[data.name == top_name].группа.describe()["top"]
    top_fclt: str = data[data.name == top_name][data.группа == top_group].факультет.describe()["top"]
    top_course: int = data[data.name == top_name][data.группа == top_group][data.факультет == top_fclt].курс.describe()[
        "top"
    ]
    frac = round(data.name.value_counts(normalize=True)[top_name], 2)

    return top_name, top_group, top_fclt, top_course, frac


# Задача 7
def find_students_with_name_starting_P(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит студентов, чье имя встречается ровно один раз и начинается на "П". Выводит их ФИО, факультет и курс.
    """
    name_pat = r"\bП[А-Яа-яё\-–]+\b"
    names = data.name.value_counts()
    ones = [name for name in names.index if re.match(name_pat, name) and names[name] == 1]
    rare_fio = pd.Series(pd.concat([data[data.name == one].фио for one in ones], axis=0))
    rare_fclt = pd.Series(pd.concat([data[data.name == one].факультет for one in ones], axis=0))
    rare_crs = pd.Series(pd.concat([data[data.name == one].курс for one in ones], axis=0))
    rare = pd.concat([rare_fio, rare_fclt, rare_crs], axis=1)

    return rare


# Задача 8
def highest_avg_grade_faculty(data: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Находит факультет, на котором средний балл студентов третьего курса самый высокий.
    Определяет пол, средний балл которого выше.
    Сначала возвращает факультет, затем пол, затем балл.
    """
    crs_stat = data[data.курс == "3-й"]
    fclts = crs_stat.факультет.unique()
    top_point_fclt_kw: Dict[str, float] = {
        fclt: crs_stat[crs_stat.факультет == fclt].средний_балл.describe()["mean"] for fclt in fclts
    }

    top_point_fclt = max(top_point_fclt_kw, key=top_point_fclt_kw.get)

    crs_stat["пол"] = crs_stat[crs_stat.факультет == top_point_fclt].patronim.apply(gender_identification)

    crs_stat_m = crs_stat[crs_stat.факультет == top_point_fclt][crs_stat.пол == "male"].средний_балл.describe()["mean"]

    crs_stat_w = crs_stat[crs_stat.факультет == top_point_fclt][crs_stat.пол == "female"].средний_балл.describe()[
        "mean"
    ]

    if crs_stat_m > crs_stat_w:
        return top_point_fclt, "male", round(crs_stat_m)

    return top_point_fclt, "female", round(crs_stat_w)


# Задача 9
def find_consecutive_students(data: pd.DataFrame) -> pd.DataFrame:
    """
    Находит первых 5 студентов, которым номера были присвоены подряд.
    Выводит их ФИО, факультет, курс и номер группы.
    """
    isu = data.sort_values(by="ису", ascending=True)

    isu["diff"] = isu["ису"].diff()
    isu["area"] = (isu["diff"] != 1).cumsum()
    area_sizes = isu.groupby("area").size()
    valid_areas = area_sizes[area_sizes >= 5].index
    firsts = isu[isu["area"].isin(valid_areas)]

    return firsts[["фио", "факультет", "курс", "группа", "ису"]].head()


if __name__ == "__main__":
    data = pd.read_csv("isu_fake_data.csv")

    names_1 = data["фио"].str.split(" ", expand=True)
    names_1 = names_1.rename(columns={0: "фамилия", 1: "имя", 2: "отчество", 3: "2е имя", 4: "3е имя", 5: "4е имя"})
    names_1["отчество"] = names_1.apply(
        lambda row: " ".join(filter(None, [row["отчество"], row["2е имя"], row["3е имя"], row["4е имя"]])), axis=1
    )

    data["surname"] = names_1["фамилия"]
    data["name"] = names_1["имя"]
    data["patronim"] = names_1["отчество"]

    # Задача 1
    num_students, num_groups, fsuir = filter_fsuir_students(data)
    print(f"Студентов на ФСУиР: {num_students}, Групп: {num_groups}")

    # Задача 2
    has_homonyms, total_homonyms, homonyms_per_course, max_homonym_group = find_homonymous_students(fsuir)
    print(f"Есть однофамильцы: {has_homonyms}, Всего: {total_homonyms}, Группа с максимумом: {max_homonym_group}")
    print(f"На каждом курсе: {homonyms_per_course}")

    # Задача 3
    students_without_patronym, gender_counts = analyze_patronyms(fsuir)
    print(f"Студентов без отчества: {students_without_patronym}")
    print("Распределение по полу:", gender_counts)

    # Задача 4
    faculty_counts, max_faculty, min_faculty = faculty_statistics(data)
    print(f"Факультет с наибольшим числом студентов: {max_faculty}")
    print(f"Факультет с наименьшим числом студентов: {min_faculty}")

    # Задача 5
    mean_students, median_students = course_statistics(data)
    print("Среднее число студентов на курсах:", mean_students)
    print("Медианное число студентов на курсах:", median_students)

    # Задача 6
    popular_name, name_group, faculty, course, name_ratio = most_popular_name(data)
    print(f"Самое популярное имя: {popular_name}, Группа: {name_group}, Факультет: {faculty}, Курс: {course}")
    print(f"Доля студентов с этим именем: {name_ratio}")

    # Задача 7
    result_7 = find_students_with_name_starting_P(data)
    print("Студенты с именем, начинающимся на П и встречающимся ровно один раз:")
    print(result_7)

    # Задача 8
    fac, best_gender, best_grade = highest_avg_grade_faculty(data)
    print(f"Факультет с высоким средним баллом 3-го курса: {fac}")
    print(f"Пол с наивысшим средним баллом: {best_gender}, Средний балл: {best_grade}")

    # Задача 9
    result_9 = find_consecutive_students(data)
    print("Первые 5 студентов с подряд идущими табельными номерами:")
    print(result_9)

    # Задача на защиту
    # Для каждого факультета выведите количество групп и среднее количество студентов в группе.
    groups_numb = data.groupby(['факультет'])['группа'].nunique()
    faculties = data['факультет'].unique()
    stud_av = data.groupby(['факультет'])['группа'].count()/data.groupby(['факультет'])['группа'].nunique()

    print(f'количество групп на каждом факультете:')
    print(groups_numb)
    print(f'количество студентов в каждой группе на каждом факультете:')
    print(stud_av)

    # Для каждого факультета найдите курс, на котором обучается больше всего студентов. И курс, на котором студентов меньше всего.
    df_stud_q = pd.DataFrame(data.groupby(['факультет', 'курс']).size().reset_index(name='количество студентов'))
    min_fac_courses = df_stud_q.loc[df_stud_q.groupby(['факультет'])['количество студентов'].idxmin()]
    max_fac_courses = df_stud_q.loc[df_stud_q.groupby(['факультет'])['количество студентов'].idxmax()]

    print(f'факультеты и курсы с количесвтом студентов:')
    print(df_stud_q)

    print(f'курсы с наименьшим количеством студентов на каждом факультете:')
    print(min_fac_courses)

    print(f'курсы с наибольшим количеством студентов на каждом факультете:')
    print(max_fac_courses)
