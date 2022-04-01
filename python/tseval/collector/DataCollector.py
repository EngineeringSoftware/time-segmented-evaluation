import collections
import datetime
import tempfile
import traceback
from pathlib import Path
from typing import *

from seutil import BashUtils, GitHubUtils, IOUtils, LoggingUtils, TimeoutException, TimeUtils
from seutil.project import Project
from tqdm import tqdm

from tseval.data.MethodData import MethodData
from tseval.data.RevisionIds import RevisionIds
from tseval.Environment import Environment
from tseval.Macros import Macros
from tseval.Utils import Utils

logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)


class DataCollector:

    def __init__(
            self,
    ):
        self.repos_dir = Macros.results_dir / "repos"
        self.repos_dir.mkdir(parents=True, exist_ok=True)

        self.repos_downloads_dir: Path = Macros.repos_downloads_dir
        IOUtils.mk_dir(self.repos_downloads_dir)
        Project.set_downloads_dir(self.repos_downloads_dir)

        self.raw_data_dir = Macros.raw_data_dir
        self.work_dir = Macros.work_dir
        return

    def search_github_java_repos(self):
        indexed_repos = {}
        logs = {
            "start_time": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            "num_repo_topic": 0,
            "num_repo_user": [],
        }

        # Find top 1000 starred repos under Java topic
        repos_topic = GitHubUtils.search_repos(q="topic:java language:java stars:>=20", sort="stars", order="desc", max_num_repos=1000)
        logs["num_repo_topic"] = len(repos_topic)
        for repo in repos_topic:
            indexed_repos[repo.clone_url] = repo

        # Find top starred repos per user, list taken from https://github.com/collections/open-source-organizations
        #   (on Dec 15 2020) union {apache, facebook}
        for user in [
            "adobe", "RedHatOfficial", "cfpb", "Netflix", "Esri", "square", "twitter",
            "gilt", "guardian", "Yelp", "Shopify", "SAP", "IBM", "microsoft", "artsy",
            "OSGeo", "godaddy", "cloudflare", "eleme", "didi", "alibaba", "google",
            "proyecto26", "mozilla", "zalando", "stripe", "newrelic", "docker",
            "ExpediaGroup", "apache", "facebook",
        ]:
            repos_user = GitHubUtils.search_repos(q=f"language:java user:{user} stars:>=20")
            logs["num_repo_user"].append((user, len(repos_user)))
            for repo in repos_user:
                indexed_repos[repo.clone_url] = repo

        projects = []
        tbar = tqdm(desc=f"Processing...", total=len(indexed_repos))
        for repo in indexed_repos.values():
            # Put together a project instance
            p = Project()
            p.url = GitHubUtils.ensure_github_api_call(lambda g: repo.clone_url)
            p.data["user"] = GitHubUtils.ensure_github_api_call(lambda g: repo.owner.login)
            p.data["repo"] = GitHubUtils.ensure_github_api_call(lambda g: repo.name)
            p.full_name = f"{p.data['user']}_{p.data['repo']}"
            p.data["branch"] = GitHubUtils.ensure_github_api_call(lambda g: repo.default_branch)
            p.data["stars"] = GitHubUtils.ensure_github_api_call(lambda g: repo.stargazers_count)
            p.data["languages"] = GitHubUtils.ensure_github_api_call(lambda g: repo.get_languages())
            projects.append(p)
            tbar.update(1)
        tbar.close()

        IOUtils.dump(
            self.repos_dir / "github-java-repos.json",
            IOUtils.jsonfy(projects),
            IOUtils.Format.jsonPretty,
        )
        IOUtils.dump(
            self.repos_dir / "github-java-repos-logs.json",
            logs,
            IOUtils.Format.jsonNoSort,
        )
        return

    def filter_repos(
            self,
            year_end: int = 2021,
            year_cnt: int = 3,
            loc_min: int = 1e6,
            loc_max: int = 2e6,
            star_min: int = 20,
    ):
        projects: List[Project] = IOUtils.dejsonfy(IOUtils.load(self.repos_dir / "github-java-repos.json", IOUtils.Format.json), List[Project])
        logs = {
            "start_time": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            "configs": {
                "year_end": year_end,
                "year_cnt": year_cnt,
                "loc_min": loc_min,
                "loc_max": loc_max,
                "star_min": star_min,
            },
            "num_repo_initial": len(projects),
            "filtered_repos": [],
            "success": 0,
            "fail": 0,
        }

        tbar = tqdm("Filtering... (+0 -0)")
        tbar.reset(len(projects))
        success = 0
        fail = 0
        filtered_projects = []
        for p in projects:
            tbar.set_description(f"Filtering: {p.full_name} (+{success} -{fail})")
            filtered = True
            if p.data["stars"] < star_min:
                logs["filtered_repos"].append((p.full_name, "star less than min"))
            elif p.data["languages"].get("Java") < loc_min:
                logs["filtered_repos"].append((p.full_name, "loc less than min"))
            elif p.data["languages"].get("Java") > loc_max:
                logs["filtered_repos"].append((p.full_name, "loc greater than max"))
            else:
                try:
                    # Try to download the project
                    with TimeUtils.time_limit(300):
                        p.clone()
                    p.checkout(p.data["branch"], is_forced=True)

                    # Find out the shas at each year
                    p.data["years"] = {}
                    with IOUtils.cd(p.checkout_dir):
                        for year in range(year_end, year_end-year_cnt-1, -1):
                            rr = BashUtils.run(f'git rev-list HEAD --first-parent --pretty="%at" --after="{year-1}-1-1" --before="{year}-1-1" -1')
                            if rr.return_code == 0 and len(rr.stdout.strip()) > 0:
                                _, sha, timestamp = rr.stdout.split()
                                p.data["years"][year] = [sha, timestamp]
                            else:
                                logs["filtered_repos"].append((p.full_name, f"no sha on year {year}"))
                                break

                            if year == year_end - year_cnt:
                                filtered = False
                except KeyboardInterrupt:
                    raise
                except TimeoutException:
                    logs["filtered_repos"].append((p.full_name, "cannot clone"))
                except:
                    logger.warning(f"Project {p.full_name} failed: {traceback.format_exc()}")
                    logs["filtered_repos"].append((p.full_name, traceback.format_exc()))
                finally:
                    IOUtils.rm_dir(p.checkout_dir)

            if not filtered:
                success += 1
                filtered_projects.append(p)
            else:
                fail += 1
            tbar.update(1)

        tbar.close()
        logs["success"] = success
        logs["fail"] = fail

        IOUtils.dump(
            self.repos_dir / "filtered-repos.json",
            IOUtils.jsonfy(filtered_projects),
            IOUtils.Format.jsonPretty,
        )
        IOUtils.dump(
            self.repos_dir / "filtered-repos-logs.json",
            logs,
            IOUtils.Format.jsonNoSort,
        )
        return

    def collect_raw_data_projects(
            self,
            year_end: int = 2021,
            year_cnt: int = 3,
            skip_collected: bool = False,
            project_names: Optional[List[str]] = None,
    ):
        # Load (filtered) projects
        projects = IOUtils.dejsonfy(IOUtils.load(self.repos_dir / "filtered-repos.json", IOUtils.Format.json), List[Project])

        if project_names is not None:
            projects = [p for p in projects if p.full_name in project_names]

        tbar = tqdm(total=len(projects))
        for pi, prj in enumerate(projects):
            tbar.set_description(f"Processing {prj.full_name}")
            try:
                self.collect_raw_data_project(
                    prj,
                    year_end=year_end,
                    year_cnt=year_cnt,
                    skip_collected=skip_collected,
                )
            except KeyboardInterrupt:
                raise
            except:
                logger.warning(f"Project {prj.full_name} failed! Exception was:\n{traceback.format_exc()}")
            finally:
                tbar.update(1)
        tbar.close()

    def collect_raw_data_project(
            self,
            prj: Project,
            year_end: int = 2021,
            year_cnt: int = 3,
            skip_collected: bool = False,
    ):
        Environment.require_collector()

        prj_raw_data_dir = self.raw_data_dir / prj.full_name
        if skip_collected and prj_raw_data_dir.is_dir():
            logger.info(f"Project {prj.full_name} already collected, skipping")

        # Clean up target directory
        IOUtils.rm_dir(prj_raw_data_dir)
        prj_raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Clone repo
        with TimeUtils.time_limit(300):
            prj.clone()

        # Call Java collector
        years = list(reversed(range(year_end, year_end-year_cnt-1, -1)))
        year2sha = {y: prj.data["years"][str(y)][0] for y in years}
        config = {
            "projectDir": str(prj.checkout_dir),
            "outputDir": str(prj_raw_data_dir),
            "logFile": str(prj_raw_data_dir / "log.txt"),
            "shas": " ".join(year2sha.values()),
        }
        config_file = Path(tempfile.mktemp(prefix="tseval"))
        IOUtils.dump(config_file, config)

        logger.info(f"Starting the Java collector; check log at {prj_raw_data_dir}/log.txt")
        rr = BashUtils.run(f"java -cp {Environment.collector_jar} org.tseval.MethodDataCollector {config_file}", expected_return_code=0)
        if rr.stderr:
            logger.warning(f"STDERR of Java collector:\n{rr.stderr}")

        # Remove temp files and the repo
        IOUtils.rm_dir(prj.checkout_dir)

    def process_raw_data(
            self,
            year_end: int = 2021,
            year_cnt: int = 3,
    ):
        # Group all projects' data into work_dir
        shared_data_dir = self.work_dir / "shared"
        IOUtils.rm_dir(shared_data_dir)
        shared_data_dir.mkdir(parents=True)

        # Load (filtered) projects
        projects = IOUtils.dejsonfy(IOUtils.load(self.repos_dir / "filtered-repos.json", IOUtils.Format.json), List[Project])

        data_id = 0
        filtered_counters: Dict[str, int] = collections.defaultdict(int)
        tbar = tqdm(total=len(projects))
        for prj in projects:
            tbar.set_description(f"Processing {prj.full_name} (total collected {data_id})")
            # Read collector's results
            prj_raw_data_dir = self.raw_data_dir / prj.full_name
            if not prj_raw_data_dir.is_dir():
                logger.warning(f"Missing raw data from project {prj.full_name}")
                tbar.update(1)
                continue

            prj_method_data_list: List[MethodData] = IOUtils.dejsonfy(IOUtils.load(prj_raw_data_dir / "method-data.json", IOUtils.Format.json), List[MethodData])
            prj_revisions_ids: List[RevisionIds] = IOUtils.dejsonfy(IOUtils.load(prj_raw_data_dir / "revision-ids.json", IOUtils.Format.json), List[RevisionIds])
            prj_filtered_counters = IOUtils.load(prj_raw_data_dir / "filtered-counters.json", IOUtils.Format.json)

            # Extend dataset
            sha2ids = {ri.revision: ri.method_ids for ri in prj_revisions_ids}

            year2ids = {}
            for year in range(year_end, year_end-year_cnt-1, -1):
                if str(year) not in prj.data["years"]:
                    logger.warning(f"Missing year {year} for {prj.full_name}")
                else:
                    sha, _ = prj.data["years"][str(year)]
                    year2ids[year] = sha2ids[sha]
            id2years = collections.defaultdict(set)
            for y, ids in year2ids.items():
                for i in ids:
                    id2years[i].add(y)

            # Remove seen data (code+comment+qcname)
            seen_data: Dict[Tuple[str, str, str], int] = {}
            for md in prj_method_data_list:
                key = (md.code, md.comment, md.qcname)
                if key in seen_data:
                    id2years[seen_data[key]].update(id2years[md.id])
                else:
                    seen_data[key] = md.id

            unique_ids = set(seen_data.values())
            unique_ds = []
            for md in prj_method_data_list:
                if md.id not in unique_ids:
                    continue

                md.years = list(sorted(id2years[md.id]))
                md.prj = prj.full_name
                # Reassign id
                md.id = data_id
                data_id += 1
                unique_ds.append(md)

            MethodData.save_dataset(unique_ds, shared_data_dir, append=True)

            # Update filtered counters
            for k, v in prj_filtered_counters.items():
                filtered_counters[k] += v

            tbar.update(1)

        # Save filtered counters
        IOUtils.dump(shared_data_dir / "filtered-counters.json", filtered_counters, IOUtils.Format.jsonPretty)

        print(Utils.suggest_dvc_add(shared_data_dir))
