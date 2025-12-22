import requests
from bs4 import BeautifulSoup
import json
import time
import re


class IUEScraper:
    def __init__(self):
        self.base_url = "https://ects.ieu.edu.tr/new/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        self.all_courses = []

    def build_department_urls(self, sections):
        base = "https://ects.ieu.edu.tr/new/akademik.php"
        return {
            name: f"{base}?section={section}&sid=curr"
            for name, section in sections.items()
        }

    def get_course_links(self, dept_url):
        print(f"\n--- Bölüm Sayfası İnceleniyor: {dept_url}")
        try:
            response = requests.get(dept_url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            # syllabus linklerini yakala
            for a in soup.find_all('a', href=re.compile(r'syllabus')):
                href = a.get('href', '')
                full_link = href if href.startswith('http') else self.base_url + href
                if full_link not in links:
                    links.append(full_link)
            return links
        except Exception as e:
            print(f"HATA: Bölüm linkleri alınamadı: {e}")
            return []

    def parse_curriculum(self, dept_url):
        response = requests.get(dept_url, headers=self.headers, timeout=15)
        soup = BeautifulSoup(response.content, "html.parser")

        curriculum_map = {}

        current_year = None
        current_semester_name = None
        current_semester_number = None

        for row in soup.find_all("tr"):
            title_cell = row.find("td", class_="title")
            if title_cell:
                title_text = title_cell.get_text(strip=True)

                match = re.match(r"(\d+)\.\s*Year\s*(Fall|Spring)\s*Semester", title_text, re.IGNORECASE)
                if match:
                    current_year = int(match.group(1))
                    sem_name = match.group(2).lower()

                    if sem_name == "fall":
                        current_semester_number = (current_year - 1) * 2 + 1
                        current_semester_name = "FALL"
                    else:
                        current_semester_number = (current_year - 1) * 2 + 2
                        current_semester_name = "SPRING"
                continue

            cells = row.find_all("td")
            if len(cells) >= 3 and current_year and current_semester_number:
                course_code = cells[0].get_text(strip=True)

                local_credit = cells[-2].get_text(strip=True)
                ects = cells[-1].get_text(strip=True)

                table = row.find_parent("table")
                course_type = "elective" if table and "elective" in table.get("class", []) else "mandatory"

                curriculum_map[course_code] = {
                    "year": current_year,
                    "semester": current_semester_number,
                    "semester_name": current_semester_name,
                    "ects": int(ects) if ects.isdigit() else None,
                    "local_credit": int(local_credit) if local_credit.isdigit() else None,
                    "type": course_type
                }

        return curriculum_map

    def scrape_course_details(self, url, department_name):
        try:
            if 'lang=en' not in url:
                url += '&lang=en'

            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            course_data = {
                "department": department_name,
                "course_code": "",
                "course_name": "",
                "semester": None,
                "type": "",
                "ects": None,
                "local_credit": None,
                "objectives": "",
                "description": "",
                "prerequisites": "None",
                "weekly_topics": [],
                "year": None,
            }

            # 1. TEMEL BİLGİLER (Code, Semester, ECTS vb. içeren id="my_table")
            # Bu tablo yatay bir yapıdadır: THEAD başlıklar, TBODY veriler
            basic_info_table = soup.find('table', id='my_table')
            if basic_info_table:
                headers = [th.get_text(strip=True) for th in basic_info_table.find('thead').find_all('th')]
                values = [td.get_text(strip=True) for td in basic_info_table.find('tbody').find_all('td')]

                info_map = dict(zip(headers, values))
                course_data["course_code"] = info_map.get("Code", "")
                # semester normalization
                semester_raw = info_map.get("Semester", "").strip()

                semester_map = {
                    "FALL": 1,
                    "SPRING": 2
                }

                if semester_raw.isdigit():
                    semester_num = int(semester_raw)
                elif semester_raw.upper() in semester_map:
                    # fallback if only FALL / SPRING is given (cannot infer year)
                    semester_num = None
                else:
                    semester_num = None

                course_data["semester"] = semester_num

                # ECTS numeric normalization
                ects_raw = info_map.get("ECTS", "")
                course_data["ects"] = int(ects_raw) if ects_raw.isdigit() else None
                # Local credit numeric normalization
                credit_raw = info_map.get("Local Credits") or info_map.get("Credits") or ""
                course_data["local_credit"] = int(credit_raw) if credit_raw.isdigit() else None

            # 2. DERS ADI (H1 veya ilk tablodan)
            # Genelde <h1> içinde yazar veya ilk tablonun 2. hücresindedir
            h1_text = soup.find('h1')
            if h1_text:
                course_data["course_name"] = re.sub(r'\s+', ' ', h1_text.get_text()).replace(
                    "COURSE INTRODUCTION AND APPLICATION INFORMATION", "").strip()

            # Eğer H1 boşsa tablodan dene
            if not course_data["course_name"]:
                name_td = soup.find('td', string=re.compile("Course Name"))
                if name_td:
                    course_data["course_name"] = re.sub(r'\s+', ' ', name_td.find_next_sibling('td').get_text()).strip()

            # 3. DETAYLI METİNLER (Objectives, Description, Prerequisites)
            # Bu kısımlar dikey tablolardadır: TD(Label) -> TD(Value)
            labels_map = {
                "Course Objectives": "objectives",
                "Course Description": "description",
                "Prerequisites": "prerequisites"
            }

            for label, key in labels_map.items():
                label_cell = soup.find(['td', 'th', 'strong'], string=re.compile(label, re.IGNORECASE))
                if label_cell:
                    # En yakın TD'yi bul ve bir sonrakini al
                    parent_td = label_cell.find_parent('td') if label_cell.name != 'td' else label_cell
                    if parent_td and parent_td.find_next_sibling('td'):
                        course_data[key] = re.sub(r'\s+', ' ', parent_td.find_next_sibling('td').get_text()).strip()

            # 4. HAFTALIK KONULAR (id="weeks" tablosu)
            weeks_table = soup.find('table', id='weeks')
            if weeks_table:
                for row in weeks_table.find_all('tr')[1:]:  # Header'ı atla
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        course_data["weekly_topics"].append({
                            "week": re.sub(r'\s+', ' ', cols[0].get_text()).strip(),
                            "topic": re.sub(r'\s+', ' ', cols[1].get_text()).strip()
                        })

            return course_data
        except Exception as e:
            print(f"HATA: {url} parse edilemedi: {e}")
            return None

    def run(self, departments):
        for dept_name, url in departments.items():
            curriculum_map = self.parse_curriculum(url)
            course_links = self.get_course_links(url)
            print(f"--- {dept_name}: {len(course_links)} ders bulundu. İşleniyor...")

            for link in course_links:
                # Linkten ders kodunu tahmin et (Tekrar kontrolü için hızlı yöntem)
                code_match = re.search(r'course_code=([^&]+)', link)
                if not code_match: continue

                details = self.scrape_course_details(link, dept_name)

                if details:
                    code = details.get("course_code")
                    if code in curriculum_map:
                        meta = curriculum_map[code]
                        details["year"] = meta["year"]
                        details["semester"] = meta["semester"]
                        details["semester_name"] = meta["semester_name"]
                        details["ects"] = meta["ects"]
                        details["local_credit"] = meta["local_credit"]
                        details["type"] = meta["type"]

                if details and details["course_code"] and details["course_code"] != "N/A":
                    real_code = details["course_code"]
                    self.all_courses.append(details)
                    print(f"BAŞARILI: {real_code} - {details['course_name'][:30]}...")

                time.sleep(0.1)

        # JSON Kayıt
        with open('iue_curriculum.json', 'w', encoding='utf-8') as f:
            json.dump(self.all_courses, f, ensure_ascii=False, indent=4)

        print(f"\nİŞLEM TAMAM! Toplam {len(self.all_courses)} ders kaydedildi.")


if __name__ == "__main__":
    DEPARTMENT_SECTIONS = {
        "Software Engineering": "se.cs.ieu.edu.tr",
        "Computer Engineering": "ce.cs.ieu.edu.tr",
        "Industrial Engineering": "is.cs.ieu.edu.tr",
        "Electrical and Electronics Engineering": "ete.cs.ieu.edu.tr"
    }

    scraper = IUEScraper()
    target_depts = scraper.build_department_urls(DEPARTMENT_SECTIONS)
    scraper.run(target_depts)