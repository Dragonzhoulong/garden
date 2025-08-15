import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"
import { SimpleSlug } from "./quartz/util/path"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [
    Component.Comments({
      provider: 'giscus',
      options: {
        // from data-repo
        repo: "Dragonzhoulong/garden",
        // from data-repo-id
        repoId: 'R_kgDOOqWeRA',
        // from data-category
        category: 'Announcements',
        // from data-category-id
        categoryId: 'DIC_kwDOOqWeRM4CuMsA',
        // from data-lang
        mapping: "pathname",
        inputPosition: "top",
        theme: "preferred_color_scheme"

        }
    }),
  ],  
    footer: Component.Footer({
    links: {
      GitHub: "https://github.com/Dragonzhoulong",
    },
  }),
}

const left = [
  Component.PageTitle(),
  Component.MobileOnly(Component.Spacer()),
  Component.Flex({
    components: [
      {
        Component: Component.Search(),
        grow: true,
      },
      { Component: Component.Darkmode() },
    ],
  }),
  Component.DesktopOnly(
    Component.RecentNotes({
      title: "Recent Writing",
      limit: 3,
      filter: (f) =>
        f.slug!.startsWith("Clippings/") &&
        f.slug! !== "Clippings/index" &&
        !f.frontmatter?.noindex,
      linkToMore: "Clippings/" as SimpleSlug,
    }),
  ),
  Component.DesktopOnly(
    Component.RecentNotes({
      title: "Recent posts",
      limit: 3,
      filter: (f) => f.slug!.startsWith("posts/"),
      linkToMore: "posts/" as SimpleSlug,
    }),
  ),
]

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [Component.ArticleTitle(), Component.ContentMeta(), Component.TagList()],
  left,
  right: [
    Component.Graph({
      localGraph: {
        showTags: true,
      },
      globalGraph: {
        showTags: true,
      },
    }),
    Component.DesktopOnly(Component.TableOfContents()),
    Component.Backlinks(),
  ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.ArticleTitle(), Component.ContentMeta()],
  left,
  right: [],
}
