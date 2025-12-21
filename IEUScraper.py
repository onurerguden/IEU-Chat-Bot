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
                "semester": "",
                "type": "",
                "ects": "",
                "local_credit": "",
                "objectives": "",
                "description": "",
                "prerequisites": "None",
                "weekly_topics": []
            }

            # 1. TEMEL BİLGİLER (Code, Semester, ECTS vb. içeren id="my_table")
            # Bu tablo yatay bir yapıdadır: THEAD başlıklar, TBODY veriler
            basic_info_table = soup.find('table', id='my_table')
            if basic_info_table:
                headers = [th.get_text(strip=True) for th in basic_info_table.find('thead').find_all('th')]
                values = [td.get_text(strip=True) for td in basic_info_table.find('tbody').find_all('td')]

                info_map = dict(zip(headers, values))
                course_data["course_code"] = info_map.get("Code", "")
                course_data["semester"] = info_map.get("Semester", "")
                course_data["ects"] = info_map.get("ECTS", "")
                course_data["local_credit"] = (
                    info_map.get("Local Credits") or
                    info_map.get("Credits") or
                    ""
                )

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
                "Course Type": "type",
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
            course_links = self.get_course_links(url)
            print(f"--- {dept_name}: {len(course_links)} ders bulundu. İşleniyor...")

            for link in course_links:
                # Linkten ders kodunu tahmin et (Tekrar kontrolü için hızlı yöntem)
                code_match = re.search(r'course_code=([^&]+)', link)
                if not code_match: continue

                details = self.scrape_course_details(link, dept_name)

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
    target_depts = {
        "Software Engineering": "https://ects.ieu.edu.tr/new/akademik.php?section=se.cs.ieu.edu.tr&sid=curr",
        "Computer Engineering": "https://ects.ieu.edu.tr/new/akademik.php?section=ce.cs.ieu.edu.tr&sid=curr",
        "Industrial Engineering": "https://ects.ieu.edu.tr/new/akademik.php?section=is.cs.ieu.edu.tr&sid=curr",
        "Electrical and Electronics Engineering": "https://ects.ieu.edu.tr/new/akademik.php?section=ete.cs.ieu.edu.tr&sid=curr"
    }

    scraper = IUEScraper()
    scraper.run(target_depts)