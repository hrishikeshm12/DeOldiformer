import pkg_resources

# List all installed libraries in the current environment
installed_packages = pkg_resources.working_set
installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])

# Write the list to requirements.txt
with open('requirements.txt', 'w') as f:
    for package in installed_packages_list:
        f.write(package + '\n')
