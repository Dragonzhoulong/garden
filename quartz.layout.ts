import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"
import { SimpleSlug } from "./quartz/util/path"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [],
  footer: Component.Footer({
    links: {
      GitHub: "https://github.com/jackyzha0",
      Twitter: "https://twitter.com/_jzhao",
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
      title: "Recent Clippings",
      limit: 4,
      filter: (f) =>
        f.slug!.startsWith("Clippings/") && f.slug! !== "posts/index" && !f.frontmatter?.noindex,
      linkToMore: "posts/" as SimpleSlug,
    }),
  ),
  Component.DesktopOnly(
    Component.RecentNotes({
      title: "Recent Notes",
      limit: 2,
      filter: (f) => f.slug!.startsWith("thoughts/"),
      linkToMore: "thoughts/" as SimpleSlug,
    }),
  ),
]

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
  ],
  left,
  right: [
    Component.Graph({
      localGraph: {
        showTags: false,
      },
      globalGraph: {
        showTags: false,
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
