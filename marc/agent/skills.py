import asyncio
from pathlib import Path
from typing import Any, final, override

import yaml
from aiofiles import open as async_open
from anyio import Path as AsyncPath
from pydantic import Field
from pydantic.dataclasses import dataclass
from textual import log


@dataclass
class Skill:
    # Follows the specification at https://agentskills.io/specification

    location: Path

    name: str = Field(max_length=64)
    description: str = Field(max_length=1024)
    license: str | None = Field(default=None)
    compatibility: str | None = Field(default=None, max_length=500)
    metadata: dict[str, Any] = Field(default={})  # pyright: ignore[reportExplicitAny]
    allowed_tools: list[str] = Field(default=[])

    @override
    def __eq__(self, value: object, /) -> bool:
        if isinstance(value, Skill):
            return self.name == value.name
        return False


@final
class SkillLoader:
    GLOBAL_SKILLS_DIR = AsyncPath(Path.home()) / ".agents" / "skills"

    @classmethod
    async def _load_skill(cls, skill_dir: AsyncPath) -> Skill | None:
        skill_dir_absolute = await skill_dir.absolute()

        skill_md = skill_dir_absolute / "SKILL.md"
        if not await skill_md.exists():
            log(f"Directory {skill_dir_absolute} does not contain SKILL.md.")
            return None

        # Read frontmatter
        async with async_open(skill_md) as f:
            frontmatter_begin = False
            frontmatter_content = []

            while line := await f.readline():
                if line.strip() == "---":
                    if not frontmatter_begin:
                        frontmatter_begin = True
                    else:
                        break
                elif frontmatter_begin:
                    frontmatter_content.append(line)

            frontmatter = yaml.safe_load("".join(frontmatter_content))

        if not isinstance(frontmatter, dict):
            log(f"Directory {skill_dir_absolute} has invalid frontmatter.")
            return None

        try:
            skill_name = frontmatter["name"]
            skill_description = frontmatter["description"]
        except KeyError:
            # Missing required properties
            log(f"Directory {skill_dir_absolute} has invalid frontmatter.")
            return None

        if skill_name != skill_dir.name:
            log(
                f"Skill {skill_name} does not match directory name at {skill_dir_absolute}, loading anyway."
            )

        return Skill(
            location=Path(skill_md),
            name=skill_name,
            description=skill_description,
            license=frontmatter.get("license"),
            compatibility=frontmatter.get("compatibility"),
            metadata=frontmatter.get("metadata", {}),
            allowed_tools=frontmatter.get("allowed_tools", []),
        )

    @classmethod
    async def _load_skills_from(cls, skills_dir: AsyncPath) -> list[Skill]:
        if not await skills_dir.exists():
            return []

        skill_dirs = skills_dir.glob("*/")
        skill_load_tasks = [cls._load_skill(d) async for d in skill_dirs]
        skills = await asyncio.gather(*skill_load_tasks)
        # skills = [await t for t in skill_load_tasks]
        skills = [s for s in skills if s]
        return skills

    @classmethod
    async def discover(cls) -> list[Skill]:
        project_skills_dir = await AsyncPath.cwd()
        project_skills = await cls._load_skills_from(project_skills_dir)

        global_skills = await cls._load_skills_from(cls.GLOBAL_SKILLS_DIR)

        all_skills: list[Skill] = []
        all_skills.extend(project_skills)
        for skill in global_skills:
            if skill in all_skills:
                log(f"Skill {skill.name} overridden by project skills.")
            else:
                all_skills.append(skill)

        return all_skills

    @classmethod
    async def load(cls, skill: Skill) -> str:
        skill_md = AsyncPath(skill.location)
        return await skill_md.read_text()
